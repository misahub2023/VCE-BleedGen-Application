import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:syncfusion_flutter_xlsio/xlsio.dart' show  Workbook, Worksheet;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as imglib;
import 'package:flutter/painting.dart';
import 'package:camera_app/main.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key}) : super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  // List? _result;
  List<List<double>>? _result;
  List<List<double>> _mask=List.generate(224, (i) => List.generate(224, (j) => 0.0));
  bool _imageSelected = false;
  bool _loading=false;
  bool _isDetected=false;
  final _imagePicker=ImagePicker();
  //---------------------------
  late tfl.Interpreter _classifier;
  late tfl.Interpreter _arch;
  late List inputShape;
  late List outputShape;
  late tfl.TensorType inputType;
  late tfl.TensorType outputType;

  //-image specs---------------------------
  double x=0;
  late double y=0;
  late double h=0;
  late double w=0;
  late double cls=0;
  late double conf=0;
  //--camera----------------------------------
  CameraImage? cameraImages;
  CameraController? cameraController;
  bool _batchPredictionsComplete = false;
  bool _loadingPredictions=false;
  //----batch detections---------------------------------------
  List<File> batch=[];
  List<List<List<double>>> _batchResults=[];
  //-------------------------------------
  //--------------------------image selection----------------------------
  @override
  Future getImage(ImageSource source) async {
    final image = await ImagePicker().pickImage(source: source);
    if (image == null) {
      return;
    }
    final imageTemporary = File(image.path);
    setState(() {
      _image = imageTemporary;
      _imageSelected = false;
      _result = null;
    });
    classifyImage(_image); //
  }

  Future getVideo(ImageSource source) async {
    final video = await ImagePicker().pickVideo(source: source);
    if (video == null) {
      return;
    }

    //frame extraction
    batch.clear();
    print("video");

    setState((){
      _image=null;
      _result=null;
    });
  }

  Future<List<File>> selectImageBatch() async {
    final List<File> selectedImages = [];
    try{
      final List<XFile> pickedImages=await ImagePicker().pickMultiImage();
      if(pickedImages !=null && pickedImages.isNotEmpty){
        for (var pickedImage in pickedImages) {
          selectedImages.add(File(pickedImage.path));
        }
      }
      print("files selected");
    } catch(e){
      print("Error");
    }
    classifyBatch(selectedImages);
    return selectedImages;
  }
  //-----------------------ML-------------------------------------------------
  //----for image---------------

  Future classifyImage(File? image) async {
    if(image==null){return;}
    final imageBytes = await image.readAsBytes();

    var inputTensor = preProcessImage(imageBytes);

    //using separate classifier
    // var outputTensor_c = List.filled(1 * 1, 0.0).reshape([1, 1]);
    //
    // _classifier.run(inputTensor,outputTensor_c);
    // print("bleeding probability : ${outputTensor_c[0][0]}");
    // if(outputTensor_c[0][0]>0.0){
    if(true){
      var outputTensor = List.filled(1 * 3087 * 6, 0.0).reshape([1, 3087, 6]);

      //running the arhcitectrue-------------------------------------------
      print("--------------------using combined architecture---------------");
      var output0 = List<double>.filled(1 * 3087 * 6, 0.0).reshape([1, 3087, 6]);
      var output1 = List<double>.filled(1 * 224 * 224 * 1, 0.0).reshape([1, 224, 224, 1]);
      var outputs = {0: output0, 1: output1};
      var inputs = [inputTensor];
      _arch.runForMultipleInputs(inputs,outputs); //---------------arc
      List<List<double>> detections = postProcess_detector(output0,0.2,0.8);
      List<List<double>> binaryMask=postProcess_segmentor(output1,0.5);
      //print(binaryMask);
      // integrated classification--------------------------------------------
      bool isDetected=false;
      double sum = 0.0;
      for (var row in binaryMask) {
        for (var element in row) {
          sum += element;
        }
      }
      double mask_percentage=(sum/50176)*100; //224x224
      double mask_percentage_threshold=0.10;
      print('mask percentage $mask_percentage');
      print('mask percentage threshold $mask_percentage_threshold');

      if(detections.isEmpty) {
        // Detections empty, check mask
        // isDetected = binaryMask.any((row) => row.any((element) => element != 0.0));

        print("Segmentor as classifier");

        if(mask_percentage <mask_percentage_threshold) {
          isDetected = false;
        }else{
          isDetected=true;
        }
        conf = 0;
      }else{
        // Detections exist, check confidence
        conf = detections[0][4];
        print("detector as classifier");
        print('conf $conf');
        isDetected = conf > 0.30;  //30% is the confidence threshold
        if(isDetected==false){
          print("segmentor helped");
          if(mask_percentage >mask_percentage_threshold) {
            isDetected = true;
          }else{
            isDetected=false;
          }
        }
      }

      //-------------------------------------------------------------------
      setState(() {
        if(detections.isEmpty){conf=0;}else{conf=detections[0][4];}
        _loading=false;
        _isDetected=isDetected;
        print(_isDetected);
        _result=detections;
        _mask=binaryMask;
      });
    }else{
      setState(() {
        _result=[];
        _isDetected=false;
        _mask=List.generate(224, (i) => List.generate(224, (j) => 0.0));
      });
    }
  }
  //-----------for batch images---------------------------------
  Future classifyBatch(List<File> batchImages) async {
    _loadingPredictions = true;
    _batchPredictionsComplete = false;
    setState(() {
      processing=true;
    });

    List<List<List<double>>> batchDetections=[];
    for(var imageFile in batchImages){
      final imageBytes=await imageFile.readAsBytes();
      var inputTensor=preProcessImage(imageBytes);
      var outputTensor=List.filled(1 * 3087 * 6, 0.0).reshape([1, 3087, 6]);
      var outputTensor_c = List.filled(1 * 1, 0.0).reshape([1, 1]);

      //using separate classifier
      // _classifier.run(inputTensor,outputTensor_c);
      // print("bleeding probability : ${outputTensor_c[0][0]}");
      // double classificationThreshold=0.3;
      // if(outputTensor_c[0][0]>classificationThreshold){

      if(true){
        var output0 = List<double>.filled(1 * 3087 * 6, 0.0).reshape([1, 3087, 6]);
        var output1 = List<double>.filled(1 * 224 * 224 * 1, 0.0).reshape([1, 224, 224, 1]);
        var outputs = {0: output0, 1: output1};
        var inputs = [inputTensor];
        _arch.runForMultipleInputs(inputs,outputs); //---------------arc
        List<List<double>> detections = postProcess_detector(output0,0.2,0.7);
        List<List<double>> binaryMask=postProcess_segmentor(output1,0.5);

        // integrated classification--------------------------------------------
        bool isDetected=false;
        if(detections.isEmpty) {
          // Detections empty, check mask
          isDetected = binaryMask.any((row) => row.any((element) => element != 0.0));
          conf = 0;
        }else{
          // Detections exist, check confidence
          conf = detections[0][4];
          isDetected = conf > 30 && binaryMask.any((row) => row.any((element) => element != 0.0));
          isDetected = conf > 30 && binaryMask.any((row) => row.any((element) => element != 0.0));
          if(isDetected==false){
            isDetected = binaryMask.any((row) => row.any((element) => element != 0.0));
          }
        }
        if(isDetected) {
          batchDetections.add(detections);
        }
        setState(() {
          if(detections.isEmpty){conf=0;}else{conf=detections[0][4];}
          _loading=false;
          _isDetected=isDetected;
          print(_isDetected);
          _result=detections;
        });
      }else{
        setState(() {
          _result=[];
          _isDetected=false;
        });
      }
    }
    //printing
    // for(List<List<double>> pred in batchDetections){
    //   print("---------$pred");
    // }
    setState((){
      _batchResults=batchDetections;
      _loadingPredictions = false;
      _batchPredictionsComplete = true;
      processing=false;

    });
  }


  //-------------------------------------------------------------------------------------------------
//--------------------------------data processing-----------------------------------------------------------------
//-------------------------------image processing---------------------------------------------------
  List<List<List<List<double>>>> preProcessImage(Uint8List imageBytes) {
    imglib.Image img = imglib.decodeImage(imageBytes)!;
    imglib.Image resizedImage = imglib.copyResize(img, width: 224, height: 224); //risizing

    List<List<List<List<double>>>> inputValues = List.generate(1, (batchIndex) {
      List<List<List<double>>> batch = [];
      for (int row = 0; row < 224; row++) {
        List<List<double>> rowValues = [];
        for (int col = 0; col < 224; col++) {
          List<double> pixelValues = [];

          int pixel = resizedImage.getPixel(col, row);
          //feature scaling
          double r = imglib.getRed(pixel)/255.0;
          double g = imglib.getGreen(pixel)/255.0;
          double b = imglib.getBlue(pixel)/255.0;

          pixelValues.add(r);
          pixelValues.add(g);
          pixelValues.add(b);

          rowValues.add(pixelValues);
        }
        batch.add(rowValues);
      }
      return batch;
    });

    return inputValues; //[1,224,224,3]
  }
  //------------------------------------------------
  List<List<double>> postProcess_detector(List<dynamic> outputTensor,double maxConfidence,double iouThreshold){
    // double maxConfidence =0.3;//detection threshold
    // double iou_threshold=0.9;// max_suppressing threshold
    List<List<double>> detections=[];
    for(int i=0;i<outputTensor[0].length;i++){
      List<dynamic> prediction=outputTensor[0][i]; // [x,y,w,h,conf,class]
      double x = prediction[0];
      double y = prediction[1];
      double w = prediction[2];
      double h = prediction[3];
      double conf = prediction[4];

      if(conf>maxConfidence){
        detections.add([x,y,w,h,conf,prediction[5]]);
      }
    }

    detections.sort((a, b) => b[4].compareTo(a[4]));
    print("detections passed the threshold ($maxConfidence) :${detections.length}");
    List<List<double>> selections = List.from(detections); // Make a copy of detections
    for (int i = 0; i < selections.length; i++) {
      for (int j = i + 1; j < selections.length; j++) {
        if (iou(selections[i], selections[j]) >= iouThreshold) {
          selections.removeAt(j); // Remove the j-th detection
          j--; // Adjust j since the list has been modified
        }
      }
    }
    print("detections passed the iou ($iouThreshold) :${selections.length}");
    return selections;
    //return detections;
  }

  double iou(List<double> a,List<double> b){
    double x1 = a[0];
    double y1 = a[1];
    double w1 = a[2];
    double h1 = a[3];

    double x2 = b[0];
    double y2 = b[1];
    double w2 = b[2];
    double h2 = b[3];

    double x1_inter=max(x1,x2);
    double y1_inter=max(y1,y2);
    double x2_inter=min(x1+w1,x2+w2);
    double y2_inter=min(y1+h1,y2+h2);

    double intersection_area=max(0,x2_inter-x1_inter)*max(0,y2_inter-y1_inter);
    double union_area=w1*h1+w2*h2;

    return intersection_area/(union_area-intersection_area);
  }

  //-------------------------------------------------------------
  List<List<double>> postProcess_segmentor(List<dynamic> outputTensor, double threshold) {
    print("segmentation threshold $threshold");
    int rows = 224;
    int columns = 224;
    List<List<double>> binaryMask = List.generate(rows, (rowIndex) {
      return List.generate(columns, (colIndex) {
        double value = outputTensor[0][rowIndex][colIndex][0];
        return value > threshold ? 1.0 : 0.0;
      });
    });
    return binaryMask;
  }
//---------------------------------------------------------------------------------------------------
  bool saving=false;
  bool processing=false;
  Future<void> saveFiles(List<List<List<double>>> batchResults) async {
    setState(() {
      saving=true;
    });
    try{
      print("saving...");
      final Workbook workbook=Workbook();
      final Worksheet sheet=workbook.worksheets[0];

      sheet.getRangeByName('A1').setText('Image');
      sheet.getRangeByName('B1').setText('X');
      sheet.getRangeByName('C1').setText('Y');
      sheet.getRangeByName('D1').setText('Width');
      sheet.getRangeByName('E1').setText('Height');
      sheet.getRangeByName('F1').setText('Confidence');
      sheet.getRangeByName('G1').setText('Class');

      int rowIndex = 2;

      for(int i=0;i<_batchResults.length;i++){
        final List<List<double>> detections=batchResults[i];
        final File imageFile = batch[i];
        final String imageName = imageFile.uri.pathSegments.last;
        for(final detection in detections){
          sheet.getRangeByIndex(rowIndex, 1).setText(imageName);
          sheet.getRangeByIndex(rowIndex,2).setValue((detection[0]));
          sheet.getRangeByIndex(rowIndex, 3).setValue(detection[1]);
          sheet.getRangeByIndex(rowIndex, 4).setValue(detection[2]);
          sheet.getRangeByIndex(rowIndex, 5).setValue(detection[3]);
          sheet.getRangeByIndex(rowIndex, 6).setValue(detection[4]*100);
          sheet.getRangeByIndex(rowIndex, 7).setValue(detection[5]);
          rowIndex++;
        }
      }
      //----------------saving----------------
      Directory? dir;
      try{
        if(Platform.isAndroid){
          if(await _requestPermission(Permission.storage)){
            dir=await getExternalStorageDirectory();
            final String timestamp = DateTime.now().toString(); // Generate timestamp
            final String fileName = 'batch_detections_$timestamp.xlsx'; // Append timestamp to file name
            final String filePath = '${dir!.path}/$fileName';
            final List<int> bytes = workbook.saveAsStream();
            final File file=File(filePath);
            await file.writeAsBytes(bytes);
            print('Excel file saved to: $filePath');
          }
        }
      } catch (e){
        print(e);
      }
    } catch(e){
      print('Error saving Excel file $e');
    }
    setState(() {
      saving=false;
    });
  }

  Future<bool> _requestPermission(Permission permission) async {
    if(await permission.isGranted){
      return true;
    }else{
      var res=await permission.request();
      if(res==PermissionStatus.granted){
        return true;
      }else{
        false;
      }
    }return false;
  }
//-------------------------------------------------------------
  // Input shape: [1, 224, 224, 3]
  // Output shape: [1, 10647, 6]

  loadCamera(){
    cameraController=CameraController(cameras![0],ResolutionPreset.medium);
    cameraController!.initialize().then((value){
      if(!mounted){
        return;
      }else{
        setState(() {
          cameraController!.startImageStream((imageStream) {
            cameraImages=imageStream;
            //classifyVideo(imageStream);
          });
        });
      }
    });
  }

  Future<void> loadModel() async {
    _classifier = await tfl.Interpreter.fromAsset("assets/classifier.tflite");

    _arch=await tfl.Interpreter.fromAsset("assets/architecture.tflite");

    print('-----------arch---------Input shape: ${_arch.getInputTensor(0).shape}');
    print('-----------arch detector-----Output shape: ${_arch.getOutputTensor(0).shape}');
    print('-----------arch-segmentor----Output shape: ${_arch.getOutputTensor(1).shape}');
  }

  void initState(){
    super.initState();
    _loading=true;
    // UserSheetsApi.init();
    loadModel().then((value){
      setState(() {
        _loading=false;
      });
    });
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('WCE BleedGen App'),
      ),
      body: Center(
        child: Column(
          children: [
            (_loadingPredictions)?
            const SizedBox(
                width: 224,
                height: 224,
                child: Center(
                    child: SizedBox(
                      width: 50,
                      height: 50,
                      child: CircularProgressIndicator(),
                    )
                ))
                :(_batchResults.isNotEmpty && _image==null) ?
            Container(
              width: 224,
              height: 224,
              child: Center(
                child: ElevatedButton(
                  onPressed: () {
                    saveFiles(_batchResults);
                  },
                  child: saving
                      ? const CircularProgressIndicator()
                      : const Text('Download Results'),
                ),
              ),
            )
                : _image != null
                ? Stack(
              children:[
                Image.file(
                  _image!,
                  width:224,
                  height: 224,
                  fit:BoxFit.cover,
                ),
                if(_result != null)
                  Positioned.fill(
                    child:CustomPaint(
                      painter: BoundingBoxPainter(
                        imageSize: const Size(224,224),
                        detection: _result!,
                        binaryMask: _mask
                      ),
                    ),
                  ),
              ],
            ): (cameraController != null && cameraController!.value.isInitialized)?
            SizedBox(
              child:CameraPreview(cameraController!),
            ): SizedBox(
              width: 224,
              height: 224,
              child: Container(),
            ),
            CustomButton('Pick from Gallery', () => getImage(ImageSource.gallery)),
            CustomButton('Open Camera', () {
              if(_image!=null){
                setState(() {
                  _batchResults.clear();
                  _image=null;
                  _result=null;
                });
              }
              loadCamera();
            }),
            Container(
              width: 280, // Set the desired width here
              child: ElevatedButton(
                onPressed: processing
                    ? null // Disable button while processing
                    : () async {
                  if (!processing) {
                    batch = await selectImageBatch();
                    if (batch.isNotEmpty) {
                      _image = null;
                      print("---------------------Selected images in batch: ${batch.length}");
                      // print(batch);
                    }
                  }
                },
                child: processing
                    ? CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                )
                    : Text("Select Batch"),
              ),
            ),
            if(_result != null)
              Text(
                _isDetected ? 'Detected' :'Not Detected',
                style: const TextStyle(fontSize: 20),
              ),
          ],
        ),
      ),
    );
  }
}

class CustomButton extends StatelessWidget {
  final String title;
  final VoidCallback onClick;

  CustomButton(this.title, this.onClick);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 280,
      child: ElevatedButton(
        onPressed: onClick,
        child: Align(
          alignment: Alignment.center,
          child: Text(title),
        ),
      ),
    );
  }
}

//--------------------bounding boxes------------------------
void drawBoundingBox(Canvas canvas, Size imageSize, List<List<double>> detections) {
  for (var detection in detections) {
    double x = detection[0];
    double y = detection[1];
    double w = detection[2];
    double h = detection[3];
    double confidence = detection[4];

    if (confidence >= 0.2) { //confidence_threshold
      // Scale the coordinates to match the image dimensions
      double imageWidth = imageSize.width;
      double imageHeight = imageSize.height;

      x *= imageWidth;
      y *= imageHeight;
      w *= imageWidth;
      h *= imageHeight;

      double left = x - w / 2;
      double top = y - h / 2;
      double right = x + w / 2;
      double bottom = y + h / 2;

      // Create a paint object to define the bounding box style
      Paint paint = Paint()
        ..color = Colors.green
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);

      //text
      TextStyle textStyle =const TextStyle(
        color: Colors.white,
        fontSize: 16.0,
        fontWeight: FontWeight.bold,
        backgroundColor: Colors.green,
      );
      TextSpan textSpan = TextSpan(
        text: '${(confidence * 100).toStringAsFixed(2)}%',
        style: textStyle,
      );
      TextPainter textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      double textX = left;
      double textY = top - 20.0;

      textPainter.paint(canvas, Offset(textX, textY));
    } else {
      print("No detections");
      break;
    }
  }
}

void drawMask(Canvas canvas,Size imageSize,List<List<double>> binaryMask){
  for(int row=0;row<binaryMask.length;row++){
    for(int col=0;col<binaryMask[row].length;col++){
      if(binaryMask[row][col]==1.0){
        double x=col.toDouble()*imageSize.width/binaryMask[row].length;
        double y=row.toDouble()*imageSize.height/binaryMask.length;

        Paint paint=Paint()
          ..color = Colors.blue.withOpacity(0.2)
          ..style = PaintingStyle.fill;

        canvas.drawCircle(Offset(x, y), 2.0, paint);
      }
    }
  }
}

class BoundingBoxPainter extends CustomPainter{
  final Size imageSize;
  final List<List<double>> detection;
  final List<List<double>> binaryMask;

  BoundingBoxPainter({
    required this.imageSize,
    required this.detection,
    required this.binaryMask,
  });

  @override
  void paint(Canvas canvas,Size size){
    drawMask(canvas,imageSize, binaryMask);
    drawBoundingBox(canvas,imageSize,detection);
  }
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}
