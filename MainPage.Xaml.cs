// Specify all the using statements which give us the access to all the APIs that we'll need 
using System;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace classifierPyTorch
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        // All the required fields declaration 
        private ImageClassifierModel modelGen;
        private ImageClassifierInput image = new ImageClassifierInput();
        private ImageClassifierOutput results;
        private StorageFile selectedStorageFile;
        private string label = "";
        private float probability = 0;
        private Helper helper = new Helper();
        public enum Labels
        {
            Plane,
            Car,
            Bird,
            Cat,
            Deer,
            Dog,
            Frog,
            Horse,
            Ship,
            Truck
        }
        public MainPage()
        {
            this.InitializeComponent();
            _ = LoadModel();

        }
        private async Task LoadModel()
        {
            // Get an access the ONNX model and save it in memory.
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/ImageClassifier.onnx"));
            // Instantiate the model. 
            modelGen = await ImageClassifierModel.CreateFromStreamAsync(modelFile);
        }
        private async void OpenFileButton_Click(object sender, RoutedEventArgs e)
        {
            if (!await GetImage())
            {
                return;
            }
            // After the click event happened and an input selected, begin the model execution. 
            // Bind the model input
            await ImageBind();
            // Model evaluation
            await Evaluate();
            // Extract the results
            ExtractResult();
            // Display the results  
            DisplayResult();
        }
        // A method to select an input image file
        private async Task<bool> GetImage()
        {
            try
            {
                // Trigger file picker to select an image file
                FileOpenPicker fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();
                if (selectedStorageFile == null)
                {
                    return false;
                }
            }
            catch (Exception)
            {
                return false;
            }
            return true;
        }
        // A method to convert and bide the input image. 
        private async Task ImageBind()
        {
            UIPreviewImage.Source = null;
            try
            {
                SoftwareBitmap softwareBitmap;
                using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                {
                    // Create the decoder from the stream
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                    // Get the SoftwareBitmap representation of the file in BGRA8 format
                    softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }
                // Display the image 
                SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
                await imageSource.SetBitmapAsync(softwareBitmap);
                UIPreviewImage.Source = imageSource;

                // Encapsulate the image within a VideoFrame to be bound and evaluated
                VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);
                // Resize the image size to 32x32  
                inputImage = await helper.CropAndDisplayInputImageAsync(inputImage);
                // Bind the model input with image 
                ImageFeatureValue imageTensor = ImageFeatureValue.CreateFromVideoFrame(inputImage);
                image.modelInput = imageTensor;
            }
            catch (Exception e)
            {
            }
        }
        // A method to evaluate the model
        private async Task Evaluate()
        {
            results = await modelGen.EvaluateAsync(image);
        }
        // A method to extract output from the model 
        private void ExtractResult()
        {
            // Retrieve the results of evaluation
            var mResult = results.modelOutput as TensorFloat;
            // convert the result to vector format
            var resultVector = mResult.GetAsVectorView();

            probability = 0;
            int index = 0;
            // find the maximum probability
            for (int i = 0; i < resultVector.Count; i++)
            {
                var elementProbability = resultVector[i];
                if (elementProbability > probability)
                {
                    index = i;
                }
            }
            label = ((Labels)index).ToString();
        }
        private void DisplayResult()
        {
            displayOutput.Text = label;
        }



    }
}
