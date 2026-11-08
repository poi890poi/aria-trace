package org.swiftest.variantsfixture;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.graphics.Color;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Independent native UI fixture. It contains no test framework or matching code. */
public class MainActivity extends Activity {
    public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        final int variant = Math.max(0, Math.min(2, getIntent().getIntExtra("variant", 0)));
        final boolean latencyProbe = getIntent().getBooleanExtra("latency_probe", false);
        if (latencyProbe) {
            setRequestedOrientation(variant == 2 ? android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                                                : android.content.pm.ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        }
        final LinearLayout main = new LinearLayout(this);
        main.setOrientation(LinearLayout.VERTICAL);
        main.setPadding(40, 60, 40, 60);
        main.setGravity(Gravity.TOP);
        main.setBackgroundColor(variant == 1 ? Color.DKGRAY : Color.WHITE);
        final Button action = new Button(this);
        action.setText(new String[]{"ＣＯＮＴＩＮＵＥ", "繼續", "继续"}[variant]);
        action.setContentDescription("primary-action");
        action.setBackgroundTintList(android.content.res.ColorStateList.valueOf(
                variant == 1 ? Color.rgb(220, 170, 20) : Color.rgb(80, 160, 240)));
        final TextView result = new TextView(this);
        result.setTextSize(24);
        result.setContentDescription("result-label");
        final int[] clicks = {0};
        action.setOnClickListener(v -> {
            clicks[0]++;
            result.setText((variant == 0 ? "Completed" : "完成") + (latencyProbe ? " " + clicks[0] : ""));
        });
        main.addView(action);
        main.addView(result);
        setContentView(main);

        final Dialog popup = new Dialog(this);
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(variant == 1 ? LinearLayout.HORIZONTAL : LinearLayout.VERTICAL);
        panel.setPadding(32, 32, 32, 32);
        panel.setContentDescription(variant == 0 ? "notice" : "notice.v2");
        panel.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        TextView title = new TextView(this);
        title.setText(variant == 0 ? "Optional notice" : "測試通知");
        title.setTextSize(22);
        panel.addView(title);
        Button close = new Button(this);
        close.setText(new String[]{"Close", "關閉", "关闭"}[variant]);
        close.setOnClickListener(v -> {
            popup.dismiss();
            main.setGravity(Gravity.TOP);
            main.setPadding(40, variant == 1 ? 420 : (latencyProbe && variant == 2 ? 220 : 720), 40, 60);
        });
        panel.addView(close);
        popup.setContentView(panel);
        popup.setCancelable(false);
        popup.show();
    }
}
