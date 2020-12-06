using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.iOS;
using Emgu.CV;
using UnityEngine.UI;
using Emgu.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Tracking;
using Emgu.CV.Face;
using System.Drawing;
using System.IO;
using System.Text.RegularExpressions;
using System;
using System.Threading;

public class Webcam : MonoBehaviour
{
    WebCamTexture webcam;
    Renderer r;
    Texture2D texture;
    public RawImage img;

    Material m;

    CascadeClassifier classifier;

    int count = 0;

    Rectangle rectangle;
    private Mutex mut = new Mutex();


    // Start is called before the first frame update
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length <= 0)
        {
            return;
        }

        for (int i = 0; i < devices.Length; i++)
        {
            print("Webcam available: " + devices[i].name);
        }

        r = this.GetComponent<Renderer>();

        webcam = new WebCamTexture(devices[0].name);

        // works
/*        img.texture = webcam;
        img.material.mainTexture = webcam;*/


        webcam.requestedHeight = 540;
        webcam.requestedWidth = 960;

        webcam.Play();

        //img.uvRect = new Rect(0.4f, 0.4f, 0.1f, 0.1f);

        Debug.Log(Application.dataPath + "/haarcascades/haarcascade_frontalface_default.xml");

        classifier = new CascadeClassifier(Application.dataPath + "/haarcascades/haarcascade_frontalface_default.xml"); //change accordingly

        rectangle = new Rectangle(new Point(0, 0), new Size(webcam.width, webcam.height));
    }

    // Update is called once per frame
    void Update()
    {
        /*        UnityEngine.Color[] c = webcam.GetPixels(0, 0, webcam.width, webcam.height);

                Texture2D t = new Texture2D(webcam.width, webcam.height);*/

        UnityEngine.Color[] c = webcam.GetPixels(rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height);

        Texture2D t = new Texture2D(rectangle.Width, rectangle.Height);

        t.SetPixels(c);
        t.Apply();

        img.material.mainTexture = t;

        count++;

        if (count > 30)
        {
            Detect();
            count = 0;
        }
    }


    void Detect()
    {
             
        Image<Bgr, byte> srcimage = new Image<Bgr, byte>(webcam.width, webcam.height);
        Image<Rgba, byte> rgbaImage = new Image<Rgba, byte>(webcam.width, webcam.height);

        Color32[] c = webcam.GetPixels32();

        var b = new byte[srcimage.Bytes.Length];

        for (int i = 0; i < b.Length; i += 3)
        {
            b[i] = c[i / 3].b;
            b[i + 1] = c[i / 3].r;
            b[i + 2] = c[i / 3].g;
        }

        srcimage.Bytes = b;
        Image<Gray, byte> grayImage = srcimage.Convert<Gray, byte>();

        Rectangle[] rectangles = classifier.DetectMultiScale(grayImage, 1.4, 1, new Size(100, 100), new Size(500, 500));


        if (rectangles.Length > 0)
        {
            double dist = Math.Pow(rectangles[0].X - rectangle.X, 2) + Math.Pow(rectangles[0].Y - rectangle.Y, 2);

            if (dist > 3000)
            {
                rectangle = rectangles[0];
            }
        }





        //Secondly, classifiers only take grayscale image so we convert our Bgr image to gray:

        /*        Finally, we call the DetectMultiScale method on our classifier:

        Rectangle[] rectangles = Classifier.DetectMultiScale(grayImage, 1.4, 0, new Size(100, 100), new Size(800, 800));
                Let’s review these parameters because you will probably need to tweak them for your system. The first parameter is the grayscale image.The second parameter is the windowing scale factor.This parameter must be greater than 1.0 and the closer it is to 1.0 the longer it will take to detect faces but there’s a greater chance that you will find all the faces. 1.4 is a good place to start with this parameter.


                The third parameter is the minimum number of nearest neighbors.The higher this number the fewer false positives you will get.If this parameter is set to something larger than 0, the algorithm will group intersecting rectangles and only return those that have overlapping rectangle greater than or equal to the minimum number of nearest neighbors.If this parameter is set to 0, all rectangles will be returned and no grouping will happen, which means the results may have intersecting rectangles for a single face.


             The last two parameters are the min and max sizes in pixels.The algorithm will start searching for faces with a window 800×800 and it will decrease the window size by the factor of 1.4 until it reaches the min size of 100×100.The bigger the range between the min and max size, the longer the algorithm will take to complete.


            The output of the DetectMultiScale function is a set of rectangles that represent where the faces are relative to the input image.
            It’s as easy as that.With just a few lines of code, you can detect where all the faces are in any image.*/
    }



}
