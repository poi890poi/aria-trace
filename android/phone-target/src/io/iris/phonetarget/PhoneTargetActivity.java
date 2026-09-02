package io.iris.phonetarget;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Choreographer;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** Exact-pixel, full-bleed calibration target driven by the host HTTP contract. */
public final class PhoneTargetActivity extends Activity implements SurfaceHolder.Callback {
    private final ExecutorService network = Executors.newSingleThreadExecutor();
    private final ExecutorService poller = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean orientationRetryScheduled = new AtomicBoolean(false);
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.DITHER_FLAG);
    private SurfaceView surface;
    private String baseUrl;
    private int revision = -1;
    private String token = "";
    private String mode = "image";
    private Bitmap bitmap;
    private int imageWidth;
    private int imageHeight;
    private int requestedCanonicalOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        Window window = getWindow();
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= 28) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.layoutInDisplayCutoutMode = Build.VERSION.SDK_INT >= 30
                    ? WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
                    : WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES;
            window.setAttributes(attributes);
        }
        lockToNaturalOrientation();
        surface = new SurfaceView(this);
        surface.setKeepScreenOn(true);
        surface.getHolder().addCallback(this);
        setContentView(surface);
        baseUrl = getIntent().getDataString();
        if (baseUrl == null || !baseUrl.startsWith("http://127.0.0.1:")) {
            throw new IllegalArgumentException("A loopback IRIS target URL is required");
        }
        if (!baseUrl.endsWith("/")) baseUrl += "/";
        hideSystemBars();
    }

    @Override public void onWindowFocusChanged(boolean focused) {
        super.onWindowFocusChanged(focused);
        if (focused) hideSystemBars();
    }

    @Override public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
        hideSystemBars();
        scheduleOrientationRetry();
    }

    private int currentDisplayRotation() {
        if (getDisplay() != null) return getDisplay().getRotation();
        return getWindowManager().getDefaultDisplay().getRotation();
    }

    private void lockToNaturalOrientation() {
        int rotation = currentDisplayRotation();
        int orientation = getResources().getConfiguration().orientation;
        boolean naturalLandscape =
                ((rotation == Surface.ROTATION_0 || rotation == Surface.ROTATION_180)
                        && orientation == Configuration.ORIENTATION_LANDSCAPE)
                || ((rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270)
                        && orientation == Configuration.ORIENTATION_PORTRAIT);
        requestedCanonicalOrientation = naturalLandscape
                ? ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                : ActivityInfo.SCREEN_ORIENTATION_PORTRAIT;
        setRequestedOrientation(requestedCanonicalOrientation);
    }

    private boolean canonicalOrientationReady() {
        return currentDisplayRotation() == Surface.ROTATION_0;
    }

    private void scheduleOrientationRetry() {
        if (!running.get() || !orientationRetryScheduled.compareAndSet(false, true)) return;
        main.postDelayed(() -> {
            orientationRetryScheduled.set(false);
            drawAndAcknowledge(revision >= 0);
        }, 100L);
    }

    private void hideSystemBars() {
        Window window = getWindow();
        if (Build.VERSION.SDK_INT >= 30) {
            window.setDecorFitsSystemWindows(false);
            WindowInsetsController controller = window.getInsetsController();
            if (controller != null) {
                controller.hide(WindowInsets.Type.systemBars());
                controller.setSystemBarsBehavior(
                        WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            }
        } else {
            window.getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
        }
    }

    @Override public void surfaceCreated(SurfaceHolder holder) {
        running.set(true);
        postTelemetry();
        poller.execute(this::pollLoop);
    }

    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        postTelemetry();
        drawAndAcknowledge(revision >= 0);
    }

    @Override public void surfaceDestroyed(SurfaceHolder holder) { running.set(false); }

    @Override protected void onDestroy() {
        running.set(false);
        poller.shutdownNow();
        network.shutdownNow();
        if (bitmap != null) bitmap.recycle();
        super.onDestroy();
    }

    private void pollLoop() {
        while (running.get()) {
            try {
                JSONObject state = new JSONObject(getText("state.json"));
                int nextRevision = state.getInt("revision");
                if (nextRevision != revision) {
                    mode = state.getString("mode");
                    token = state.optString("token", "");
                    Bitmap next = null;
                    if ("image".equals(mode)) {
                        byte[] bytes = getBytes("image.png?v=" + nextRevision);
                        next = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                        if (next == null) throw new IllegalStateException("Cannot decode target PNG");
                    }
                    final Bitmap published = next;
                    final int publishedRevision = nextRevision;
                    main.post(() -> {
                        Bitmap previous = bitmap;
                        bitmap = published;
                        imageWidth = published == null ? 0 : published.getWidth();
                        imageHeight = published == null ? 0 : published.getHeight();
                        revision = publishedRevision;
                        drawAndAcknowledge(true);
                        if (previous != null && previous != published) previous.recycle();
                    });
                }
            } catch (Exception ignored) {
                // The host may be between revisions or shutting down. Polling resumes.
            }
            try { Thread.sleep(40L); } catch (InterruptedException stop) { return; }
        }
    }

    private void drawAndAcknowledge(boolean acknowledge) {
        if (surface == null || !surface.getHolder().getSurface().isValid()) return;
        boolean orientationReady = canonicalOrientationReady();
        Canvas canvas = null;
        try {
            canvas = surface.getHolder().lockCanvas();
            if (canvas == null) return;
            if (!orientationReady) canvas.drawColor(Color.BLACK);
            else if ("black".equals(mode)) canvas.drawColor(Color.BLACK);
            else if ("white".equals(mode)) canvas.drawColor(Color.WHITE);
            else if (bitmap != null) {
                paint.setFilterBitmap(false);
                canvas.drawBitmap(bitmap, null,
                        new Rect(0, 0, canvas.getWidth(), canvas.getHeight()), paint);
            }
        } finally {
            if (canvas != null) surface.getHolder().unlockCanvasAndPost(canvas);
        }
        if (!orientationReady) {
            scheduleOrientationRetry();
        } else if (acknowledge) {
            final int paintedRevision = revision;
            final String paintedToken = token;
            Choreographer.getInstance().postFrameCallback(
                    frameTimeNanos -> postAcknowledgement(paintedRevision, paintedToken));
        }
    }

    private JSONObject surfaceTelemetry() {
        JSONObject value = new JSONObject();
        try {
            int width = surface == null ? 0 : surface.getWidth();
            int height = surface == null ? 0 : surface.getHeight();
            value.put("adapter_id", "android_native_surface");
            value.put("target_contract_version", 2);
            value.put("activity", getComponentName().flattenToShortString());
            value.put("canvas_width", width);
            value.put("canvas_height", height);
            value.put("surface_width", width);
            value.put("surface_height", height);
            value.put("image_natural_width", imageWidth);
            value.put("image_natural_height", imageHeight);
            value.put("surface_scale_x", imageWidth > 0 ? width / (double) imageWidth : 1.0);
            value.put("surface_scale_y", imageHeight > 0 ? height / (double) imageHeight : 1.0);
            value.put("fullscreen", true);
            value.put("immersive_mode", true);
            value.put("keep_screen_on", true);
            value.put("native_surface", true);
            value.put("display_rotation", currentDisplayRotation());
            value.put("requested_orientation", requestedCanonicalOrientation);
            value.put("canonical_orientation_ready", canonicalOrientationReady());
        } catch (Exception ignored) { }
        return value;
    }

    private void postTelemetry() {
        JSONObject value = surfaceTelemetry();
        network.execute(() -> postJson("telemetry", value));
    }

    private void postAcknowledgement(int paintedRevision, String paintedToken) {
        JSONObject value = surfaceTelemetry();
        try {
            value.put("revision", paintedRevision);
            value.put("token", paintedToken);
            value.put("painted", true);
            value.put("frame_time_ns", System.nanoTime());
        } catch (Exception ignored) { }
        network.execute(() -> postJson("ack", value));
    }

    private String getText(String path) throws Exception {
        return new String(getBytes(path), StandardCharsets.UTF_8);
    }

    private byte[] getBytes(String path) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(baseUrl + path).openConnection();
        connection.setConnectTimeout(1000);
        connection.setReadTimeout(1000);
        connection.setUseCaches(false);
        try (InputStream input = connection.getInputStream();
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16384];
            for (int count; (count = input.read(buffer)) >= 0; ) output.write(buffer, 0, count);
            return output.toByteArray();
        } finally { connection.disconnect(); }
    }

    private void postJson(String path, JSONObject value) {
        HttpURLConnection connection = null;
        try {
            connection = (HttpURLConnection) new URL(baseUrl + path).openConnection();
            connection.setConnectTimeout(1000);
            connection.setReadTimeout(1000);
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            byte[] body = value.toString().getBytes(StandardCharsets.UTF_8);
            try (OutputStream output = connection.getOutputStream()) { output.write(body); }
            connection.getResponseCode();
        } catch (Exception ignored) {
        } finally { if (connection != null) connection.disconnect(); }
    }
}
