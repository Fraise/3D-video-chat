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
    public RawImage img;

    //Renderer r;

    WebCamTexture webcam;
    int wcHeight = 0;
    int wcWidth = 0;
    Material m;


    // OpenCV
    CascadeClassifier classifier;

    // IPC
    Color32[] colorsBuffer = new Color32[0];
    private Mutex bufMut = new Mutex();

    Rectangle rectangle = new Rectangle();
    private Mutex rectMut = new Mutex();


    // Start is called before the first frame update
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length <= 0)
        {
            return;
        }

        //r = this.GetComponent<Renderer>();

        webcam = new WebCamTexture(devices[0].name);

        webcam.requestedHeight = 540;
        webcam.requestedWidth = 960;

        webcam.Play();

        wcHeight = webcam.height;
        wcWidth = webcam.width;

        classifier = new CascadeClassifier(Application.dataPath + "/haarcascades/haarcascade_frontalface_default.xml");

        rectangle = new Rectangle(new Point(0, 0), new Size(wcWidth, wcHeight));

        Thread openCVThread = new Thread(Detect);
        openCVThread.Start();
    }

    void Update()
    {
        bufMut.WaitOne();
        colorsBuffer = webcam.GetPixels32();
        bufMut.ReleaseMutex();

        rectMut.WaitOne();
        Rectangle r = rectangle;
        rectMut.ReleaseMutex();

        UnityEngine.Color[] c = webcam.GetPixels(r.X, r.Y, r.Width, r.Height);

        Texture2D t = new Texture2D(r.Width, r.Height);

        t.SetPixels(c);
        t.Apply();

        img.material.mainTexture = t;
    }


    void Detect()
    {
        while (true)
        {
            Thread.Sleep(30);

            bufMut.WaitOne();
            Color32[] c = colorsBuffer;
            bufMut.ReleaseMutex();

            if (c.Length <= 0)
            {
                continue;
            }

            Image<Bgr, byte> srcimage = new Image<Bgr, byte>(wcWidth, wcHeight);

            var b = new byte[srcimage.Bytes.Length];

            for (int i = 0; i < b.Length; i += 3)
            {
                b[i] = c[i / 3].b;
                b[i + 1] = c[i / 3].r;
                b[i + 2] = c[i / 3].g;
            }

            srcimage.Bytes = b;
            Image<Gray, byte> grayImage = srcimage.Convert<Gray, byte>();

            Rectangle[] rectangles = classifier.DetectMultiScale(grayImage, 1.4, 2, new Size(100, 100), new Size(500, 500));

            rectMut.WaitOne();
            if (rectangles.Length > 0)
            {
                double dist = Math.Pow(rectangles[0].X - rectangle.X, 2) + Math.Pow(rectangles[0].Y - rectangle.Y, 2);

                if (dist > 5000)
                {
                    rectangle = rectangles[0];
                }
            }
            rectMut.ReleaseMutex();
        }


        /*
         
        From: http://blogs.interknowlogy.com/2013/10/21/face-detection-for-net-using-emgucv/ 
        
        Let’s review these parameters because you will probably need to tweak them for your system.
        The first parameter is the grayscale image.The second parameter is the windowing scale factor.
        This parameter must be greater than 1.0 and the closer it is to 1.0 the longer it will take to
        detect faces but there’s a greater chance that you will find all the faces. 1.4 is a good
        place to start with this parameter.

        The third parameter is the minimum number of nearest neighbors. The higher this number the
        fewer false positives you will get. If this parameter is set to something larger than 0, 
        the algorithm will group intersecting rectangles and only return those that have overlapping
        rectangle greater than or equal to the minimum number of nearest neighbors.If this parameter
        is set to 0, all rectangles will be returned and no grouping will happen, which means the 
        results may have intersecting rectangles for a single face.


        The last two parameters are the min and max sizes in pixels. The algorithm will start searching
        for faces with a window 500×500 and it will decrease the window size by the factor of 1.4 until
        it reaches the min size of 100×100.The bigger the range between the min and max size, the longer
        the algorithm will take to complete.


        The output of the DetectMultiScale function is a set of rectangles that represent where the faces are relative to the input image.
        It’s as easy as that. With just a few lines of code, you can detect where all the faces are in any image.*/
    }



}
