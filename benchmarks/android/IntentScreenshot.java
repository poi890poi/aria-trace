import android.graphics.Bitmap;
import android.hardware.HardwareBuffer;
import android.os.IBinder;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/** Experimental Android 13 capture ablation; not a runtime backend. */
public final class IntentScreenshot {
    private byte[] previous;
    private int lastWidth, lastHeight;
    private final Method captureDisplay, asBitmap, getBuffer;
    private final Object captureArgs;
    private Bitmap previousBitmap;

    public IntentScreenshot() throws Exception {
        Class<?> control = Class.forName("android.view.SurfaceControl");
        long[] ids = (long[]) control.getMethod("getPhysicalDisplayIds").invoke(null);
        if (ids.length != 1) throw new IllegalStateException("Probe requires one physical display");
        Object token = control.getMethod("getPhysicalDisplayToken", long.class).invoke(null, ids[0]);
        Class<?> builder = Class.forName("android.view.SurfaceControl$DisplayCaptureArgs$Builder");
        Object config = builder.getConstructor(IBinder.class).newInstance(token);
        captureArgs = builder.getMethod("build").invoke(config);
        captureDisplay = control.getMethod("captureDisplay", captureArgs.getClass());
        Class<?> result = Class.forName("android.view.SurfaceControl$ScreenshotHardwareBuffer");
        asBitmap = result.getMethod("asBitmap");
        getBuffer = result.getMethod("getHardwareBuffer");
    }

    private Bitmap directImage() throws Exception {
        Object result = captureDisplay.invoke(null, captureArgs);
        if (result == null) throw new IllegalStateException("No screenshot");
        HardwareBuffer buffer = (HardwareBuffer) getBuffer.invoke(result);
        Bitmap hardware = (Bitmap) asBitmap.invoke(result);
        try {
            return hardware.copy(Bitmap.Config.ARGB_8888, false);
        } finally {
            hardware.recycle();
            buffer.close();
        }
    }

    private byte[] retainedCapture(boolean force) throws Exception {
        long start = System.nanoTime();
        Bitmap bitmap = directImage();
        long acquired = System.nanoTime();
        boolean same = !force && previousBitmap != null && bitmap.sameAs(previousBitmap);
        if (previousBitmap != null) previousBitmap.recycle();
        previousBitmap = bitmap;
        try {
            if (same) return new byte[0];
            ByteArrayOutputStream encoded = new ByteArrayOutputStream();
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG,80,encoded)) throw new IllegalStateException("JPEG failed");
            return encoded.toByteArray();
        } finally {
            System.err.println("bitmap,"+(acquired-start)/1e6+","+(System.nanoTime()-acquired)/1e6+","+same);
        }
    }

    private byte[] capture(boolean direct) throws Exception {
        long start = System.nanoTime();
        Bitmap bitmap = null;
        int width, height;
        byte[] pixels;
        if (direct) {
            bitmap = directImage();
            width = bitmap.getWidth(); height = bitmap.getHeight();
            pixels = new byte[bitmap.getByteCount()];
            bitmap.copyPixelsToBuffer(ByteBuffer.wrap(pixels));
        } else {
            Process process = new ProcessBuilder("/system/bin/screencap").start();
            try (DataInputStream raw = new DataInputStream(process.getInputStream())) {
                byte[] header = new byte[16];
                raw.readFully(header);
                ByteBuffer fields = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
                width = fields.getInt(); height = fields.getInt(); int format = fields.getInt();
                if (width <= 0 || height <= 0 || (long) width * height > 16000000 || format != 1)
                    throw new IllegalStateException("Unsupported pixels");
                pixels = new byte[width * height * 4];
                raw.readFully(pixels);
                if (raw.read() != -1 || process.waitFor() != 0) throw new IllegalStateException("Capture failed");
            } finally { process.destroy(); }
        }
        long acquired = System.nanoTime();
        boolean same = lastWidth == width && lastHeight == height && Arrays.equals(previous, pixels);
        previous = pixels; lastWidth = width; lastHeight = height;
        try {
            if (same) return new byte[0];
            if (bitmap == null) {
                bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels));
            }
            ByteArrayOutputStream encoded = new ByteArrayOutputStream();
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 80, encoded)) throw new IllegalStateException("JPEG failed");
            return encoded.toByteArray();
        } finally {
            if (bitmap != null) bitmap.recycle();
            System.err.println((direct ? "direct" : "screencap") + "," + (acquired-start)/1e6 + "," + (System.nanoTime()-acquired)/1e6 + "," + same);
        }
    }

    public static void main(String[] args) throws Exception {
        IntentScreenshot camera = new IntentScreenshot();
        DataOutputStream output = new DataOutputStream(System.out);
        int request, lastRequest = -1;
        while ((request = System.in.read()) != -1) {
            if (request != 'C' && request != 'D' && request != 'c' && request != 'd' && request != 'R' && request != 'r') break;
            if (request != lastRequest) {
                camera.previous = null;
                if (camera.previousBitmap != null) camera.previousBitmap.recycle();
                camera.previousBitmap = null;
            }
            lastRequest = request;
            if (request == 'c' || request == 'd') camera.previous = null;
            byte[] image = (request == 'R' || request == 'r') ? camera.retainedCapture(request == 'r') : camera.capture(request == 'D' || request == 'd');
            output.write(android.util.Base64.encode(image, android.util.Base64.NO_WRAP));
            output.writeByte(10);
            output.flush();
        }
    }
}
