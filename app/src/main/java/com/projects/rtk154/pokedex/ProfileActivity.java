package com.projects.rtk154.pokedex;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.google.firebase.auth.FirebaseAuth;

public class ProfileActivity extends AppCompatActivity {
    FirebaseAuth mFirebaseAuth;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_profile);
        mFirebaseAuth= FirebaseAuth.getInstance();
        Button logoutButton=(Button)findViewById(R.id.logoutButton);
        Button plantButton = (Button) findViewById(R.id.button);
        Button flowersButton = (Button) findViewById(R.id.button2);
        Button edibleButton = (Button) findViewById(R.id.button3);
        Button insectsButton = (Button) findViewById(R.id.button4);
        logoutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {//stop getting location

                mFirebaseAuth.signOut();
                finish();
                startActivity(new Intent(ProfileActivity.this,LoginActivity.class));
            }
        });
        plantButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {//stop getting location
                startActivity(new Intent(ProfileActivity.this,CaptureImage.class));
            }
        });
        flowersButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {//stop getting location

                startActivity(new Intent(ProfileActivity.this,CaptureImage.class));
            }
        });
        edibleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {//stop getting location

                startActivity(new Intent(ProfileActivity.this,CaptureImage.class));
            }
        });
        insectsButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {//stop getting location

                startActivity(new Intent(ProfileActivity.this,CaptureImage.class));
            }
        });
    }
}
