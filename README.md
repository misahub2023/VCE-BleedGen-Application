#  VCE-BleedGen-Application

## Project Overview
VCE-BleedGen application is a real-time application designed for the automatic classification of bleeding and non-bleeding frames and then further detection and segmentation of the bleeding region in the bleeding frames in a video capsule endoscopy (VCE) footage. It offers both single-frame and batch processing capabilities, delivering results with a latency of under 2 seconds. Built for on-premise use, VCEBleedGen runs entirely offline, ensuring full control over data privacy and security while performing all three tasks seamlessly. This efficient, locally-deployed solution aids in quickly identifying bleeding events, streamlining analysis and enhancing decision-making in VCE.

## Getting Started
These instructions will guide you through setting up and running the app on your local machine.

### Prerequisites
- Flutter SDK: [Installation Guide](https://flutter.dev/docs/get-started/install)
- Dart SDK: Comes with Flutter
- Android Studio or Xcode: For running the app on an emulator or physical device

### Installation 
#### (run these commands in the terminal)
1. Clone the repository:
    ```bash
    git clone https://github.com/misahub2023/FEM-LABELLER.git
    ```
2. Update flutter and its dependencies: 
    ```bash
    flutter upgrade
    ``` 
3. Check whether flutter is ready for use: 
    ```bash
    flutter doctor
    ```
    Resolve any issues shown by flutter doctor before moving to next steps

4. Install the dependencies:
    ```bash
    flutter pub get
    ```

### Running the App
1. Connect your device or start an emulator.
2. Run the app:
    ```bash
    flutter run
    ```

## Files Structure And their purpose
+ android/
    + [app/](android/app) 
        + `src/` (Source files for different build variants)
            + `debug/` 
                + `AndroidManifest.xml` (The manifest file for the debug build variant, specifying application metadata, permissions, and components)
            + `main/` (Source files for the main build variant)
                + `kotlin/com/example/camera/`
                    + `MainActivity.kt` (The main entry point for the Android application written in Kotlin)
                + `res/` (Resource files for the app, one can modify the launch screeen backgroud using multiple options available in this folder)
                    + `drawable-v21/` (Drawable resources for API level 21 and above, one can modify the launch screeen backgroud using multiple options available in this directory)
                        + `launch_background.xml` (Defines the launch screen background for the app)
                    + `mipmap-hdpi` (stores different versions of application icons and launcher images)
                    + `values-night/` 
                        + `styles.xml` (Defines styles for the app in night mode)
                    + `values/`
                        + `styles.xml` (Defines styles for the app)
                + `AndroidManifest.xml` (The manifest file for the main build variant, specifying application metadata, permissions, and components)
            + `profile` (Source files for the profile build variant)
                + `AndroidManifest.xml`
        + `build.gradle` (Build configuration file for the app module, specifying dependencies, plugins, and build settings)
    + [gradle/wrapper/](android/gradle/wrapper) (Contains Gradle wrapper files to ensure a specific version of Gradle is used)
        + `gradle-wrapper.properties` (Specifies properties for the Gradle wrapper, such as the Gradle distribution URL)
    + [build.gradle](android/build.gradle) (Top-level build configuration file for the project, specifying project-wide dependencies and build settings)
    + [gradle.properties](android/gradle.properties) (Specifies properties for the Gradle build system, such as JVM options and project properties)
    + `[settings.gradle](android/settings.gradle) (Specifies the Gradle settings for the project, including module names and build configurations)

+ ios/
    + [Flutter/](ios/Flutter) (This folder contains Flutter-specific files that are automatically generated and managed by the Flutter framework)
        + `AppFrameworkInfo.plist` (contains metadata about the Flutter framework)
    + [Runner.xcodeproj/](ios/Runner.xcodeproj) (This folder contains the Xcode project files for your IOS app)
        + `project.pbxproj` (project file that describes the targets, build configurations, and file references for your Xcode project)
    +  [Runner.xcworkspace/](ios/Runner.xcworkspace) (This folder contains the workspace settings for your Xcode project)
        + `contents.xcworkspacedata.xml` (ile defines the structure of your Xcode workspace, including references to your project and any dependencies)
    + [Runner/](ios/Runner) (This folder contains the main iOS project files, including the app's source code, assets, and configurations)
        + `Assets.xcassets/` (This folder contains the app's image assets, such as icons and launch images)
            + `AppIcon.appiconset/`
                +  `Contents.json` (A JSON file that describes the structure and properties of the icon set)
            + LaunchImage.imageset/
                +  `Contents.json` (A JSON file that describes the structure and properties of the launch image set)
                + `README.md` (explains steps to modify the launch screen)
        + `Base.lproj/` (This folder contains the storyboard files for the app's user interface)
            + `LaunchScreen.storyboard.xml` (defines the layout and properties of the app's launch screen)
            + `Main.storyboard.xml` (defines the layout and properties of the app's main interface)
        + `AppDelegate.swift` (This file contains the entry point for the iOS app and handles app lifecycle events)
        + `Info.plist` (contains configuration settings for the iOS app, such as app permissions, icons, and other metadata)
        + `Runner-Bridging-Header.h` (used to integrate Swift and Objective-C code)
+ [assets/](assets) (This folder contains the unified architecture for Classification , Object Detection and Semantic Segmentation ) 
+ lib/
    + [home.dart](lib/home.dart) (implements the functionality multi-task learning over Video Capsule endoscopy frames)
    + [main.dart](lib/main.dart) (The entry point of the Flutter application)
+ test/
    + [widget_test.dart](test/widget_test.dart) (File for testing the widgets of the app)

#### This application is developed for android devices so for IOS supported app, one must modify the files in `ios` folder according to the details of each file mentioned above.
## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## Credits
Palak Handa conceptualized the idea, participated in manuscript writing, figures, and supervised the project. The development of the application and its integration with machine learning models was done by Divyansh Nautiyal. Tanisha Singh and Anushka Saini participated in manuscript writing, and table development. Manas Dhir performed the machine learning analysis and figures. Manya Joshi helped in developing the Github repository of the application. No funding was received for this project. The application is a proof-of-concept, only developed for research purposes, which maybe used for other biomedical applications in AI. Please give appropriate credits before using it. Commercial use can be discussed at palakhanda97@gmail.com
