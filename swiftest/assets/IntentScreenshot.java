import android.graphics.Bitmap;
import android.hardware.HardwareBuffer;
import android.os.IBinder;
import android.os.Build;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.lang.reflect.Method;

/** Capture one JPEG per request and acknowledge identical frames without resending them. */
public final class IntentScreenshot {
    private byte[] previous;
    private int lastWidth, lastHeight;
    private boolean directAvailable = Build.VERSION.SDK_INT == 33;
    private Method captureDisplay, asBitmap, getBuffer;
    private Object captureArgs;
    private boolean usedDirect;
    private Bitmap previousBitmap;

    private Bitmap directBitmap() throws Exception {
        if (captureArgs == null) {
            Class<?> control = Class.forName("android.view.SurfaceControl");
            long[] ids = (long[]) control.getMethod("getPhysicalDisplayIds").invoke(null);
            if (ids.length != 1) throw new IllegalStateException("Expected one physical display");
            Object token = control.getMethod("getPhysicalDisplayToken", long.class).invoke(null, ids[0]);
            Class<?> builder = Class.forName("android.view.SurfaceControl$DisplayCaptureArgs$Builder");
            Object config = builder.getConstructor(IBinder.class).newInstance(token);
            captureArgs = builder.getMethod("build").invoke(config);
            captureDisplay = control.getMethod("captureDisplay", captureArgs.getClass());
            Class<?> result = Class.forName("android.view.SurfaceControl$ScreenshotHardwareBuffer");
            asBitmap = result.getMethod("asBitmap");
            getBuffer = result.getMethod("getHardwareBuffer");
        }
        Object result = captureDisplay.invoke(null, captureArgs);
        if (result == null) throw new IllegalStateException("No capture buffer");
        HardwareBuffer buffer = (HardwareBuffer) getBuffer.invoke(result);
        Bitmap hardware = null;
        try {
            hardware = (Bitmap) asBitmap.invoke(result);
            if (hardware == null) throw new IllegalStateException("No capture bitmap");
            Bitmap software = hardware.copy(Bitmap.Config.ARGB_8888, false);
            if (software == null) throw new IllegalStateException("Cannot read capture bitmap");
            return software;
        } finally {
            if (hardware != null) hardware.recycle();
            if (buffer != null) buffer.close();
        }
    }

    private byte[] encode(Bitmap bitmap, byte[] pixels, int width, int height) throws Exception {
        if (lastWidth == width && lastHeight == height && Arrays.equals(previous, pixels)) {
            return new byte[0];
        }
        previous = pixels;
        lastWidth = width;
        lastHeight = height;
        ByteArrayOutputStream encoded = new ByteArrayOutputStream();
        if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 80, encoded)) {
            throw new IllegalStateException("JPEG compression failed");
        }
        return encoded.toByteArray();
    }

    private byte[] capture(boolean preferDirect, boolean retainBitmap) throws Exception {
        usedDirect = false;
        if (preferDirect && directAvailable) {
            Bitmap bitmap = null;
            try {
                bitmap = directBitmap();
                if (retainBitmap) {
                    boolean same = previousBitmap != null && bitmap.sameAs(previousBitmap);
                    if (previousBitmap != null) previousBitmap.recycle();
                    previousBitmap = bitmap;
                    bitmap = null; // the retained comparison owns this bitmap
                    ByteArrayOutputStream encoded = new ByteArrayOutputStream();
                    if (!same && !previousBitmap.compress(Bitmap.CompressFormat.JPEG, 80, encoded)) {
                        throw new IllegalStateException("JPEG compression failed");
                    }
                    usedDirect = true;
                    return encoded.toByteArray();
                }
                byte[] pixels = new byte[bitmap.getByteCount()];
                bitmap.copyPixelsToBuffer(ByteBuffer.wrap(pixels));
                byte[] result = encode(bitmap, pixels, bitmap.getWidth(), bitmap.getHeight());
                usedDirect = true;
                return result;
            } catch (Exception error) {
                // Acquire again through the established path; never return a
                // held image after a failed direct acquisition.
                directAvailable = false;
                previous = null;
                if (previousBitmap != null) previousBitmap.recycle();
                previousBitmap = null;
            } finally {
                if (bitmap != null) bitmap.recycle();
            }
        }
        Process capture = new ProcessBuilder("/system/bin/screencap").start();
        Bitmap bitmap = null;
        try (DataInputStream raw = new DataInputStream(capture.getInputStream())) {
            byte[] header = new byte[16];
            raw.readFully(header);
            ByteBuffer fields = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
            int width = fields.getInt(), height = fields.getInt(), format = fields.getInt();
            if (width <= 0 || height <= 0 || (long) width * height > 16000000 || format != 1) {
                throw new IllegalArgumentException("Unsupported Android capture format");
            }
            byte[] pixels = new byte[width * height * 4];
            raw.readFully(pixels);
            if (raw.read() != -1 || capture.waitFor() != 0) {
                throw new IllegalStateException("Unexpected screen buffer size or capture failure");
            }
            if (lastWidth == width && lastHeight == height && Arrays.equals(previous, pixels)) {
                return new byte[0];
            }
            bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(pixels));
            return encode(bitmap, pixels, width, height);
        } finally {
            if (bitmap != null) bitmap.recycle();
            capture.destroy();
        }
    }

    public static void main(String[] args) throws Exception {
        IntentScreenshot camera = new IntentScreenshot();
        DataOutputStream output = new DataOutputStream(System.out);
        int request, lastRequest = -1;
        while ((request = System.in.read()) == 'C' || request == 'D' || request == 'R') {
            if (request != lastRequest) {
                camera.previous = null;
                if (camera.previousBitmap != null) camera.previousBitmap.recycle();
                camera.previousBitmap = null;
            }
            lastRequest = request;
            byte[] image = camera.capture(request != 'C', request == 'R');
            if (request != 'C') output.writeBytes(camera.usedDirect ? "D:" : "S:");
            // adb shell on Windows translates line endings. Keep the wire alphabet text-safe.
            output.write(android.util.Base64.encode(image, android.util.Base64.NO_WRAP));
            output.writeByte(10);
            output.flush();
        }
    }
}
