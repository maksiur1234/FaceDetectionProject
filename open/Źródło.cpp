#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

void displayImage(const cv::Mat& image, const std::string& windowName) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, image);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

std::string classifyGender(const cv::Mat& image) {
    // £adowanie modelu uczenia maszynowego do rozpoznawania p³ci
    cv::dnn::Net genderNet = cv::dnn::readNetFromCaffe("C:\\Users\\maksi\\OneDrive\\Pulpit\\open\\gender_deploy.prototxt",
        "C:\\Users\\maksi\\OneDrive\\Pulpit\\open\\gender_net.caffemodel");

    if (genderNet.empty()) {
        std::cout << "Blad wczytywania modelu." << std::endl;
        return "";
    }

    // Przygotowanie obrazu do klasyfikacji
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(227, 227));

    cv::Mat inputBlob = cv::dnn::blobFromImage(resizedImage, 1.0, cv::Size(227, 227), cv::Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);

    // Przekazywanie obrazu przez sieæ neuronow¹
    genderNet.setInput(inputBlob, "data");
    cv::Mat predictions = genderNet.forward("prob");

    // Okreœlanie p³ci na podstawie wyników klasyfikacji
    cv::Point maxLoc;
    double maxProb;
    cv::minMaxLoc(predictions, nullptr, &maxProb, nullptr, &maxLoc);

    std::string gender;
    if (maxLoc.x == 0) {
        gender = "Mezczyzna";
    }
    else {
        gender = "Kobieta";
    }

    return gender;
}

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // £adowanie klasyfikatora do wykrywania twarzy
    cv::CascadeClassifier faceCascade;
    faceCascade.load("C:\\Users\\maksi\\OneDrive\\Pulpit\\open\\haarcascades\\haarcascade_frontalface_default.xml");

    // Œcie¿ka do katalogu wejœciowego
    std::string inputDirectory;

    std::cout << "Podaj sciezke do katalogu wejœciowego: ";
    std::cin >> inputDirectory;

    // Liczniki
    int totalFaces = 0;
    int imagesWithoutFaces = 0;

    // Pobieranie wszystkich plików w katalogu wejœciowym
    for (const auto& entry : fs::directory_iterator(inputDirectory)) {
        // Sprawdzanie, czy plik jest obrazem (rozszerzenie pliku)
        std::string extension = entry.path().extension().string();
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
            // £adowanie zdjêcia
            cv::Mat image = cv::imread(entry.path().string());

            // Wyœwietlanie oryginalnego obrazu
            displayImage(image, "Oryginalny obraz");

            // Konwertowanie obrazu na odcienie szaroœci
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            // Wykrywanie twarzy na obrazie przy u¿yciu klasyfikatora Haar
            std::vector<cv::Rect> faces;
            faceCascade.detectMultiScale(gray, faces, 1.1, 5, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

            // Jeœli znaleziono twarze
            if (!faces.empty()) {
                // Zwiêkszanie licznika o liczbê wykrytych twarzy
                totalFaces += faces.size();

                // Dla ka¿dej wykrytej twarzy
                for (const cv::Rect& face : faces) {
                    // Wycinanie fragmentu obrazu zawieraj¹cego twarz
                    cv::Mat faceROI = image(face);

                    // Rozpoznawanie p³ci
                    std::string gender = classifyGender(faceROI);

                    // Rysowanie prostok¹tu wokó³ twarzy i wyœwietlanie informacji o p³ci
                    cv::rectangle(image, face, cv::Scalar(0, 255, 0), 2);
                    cv::putText(image, gender, cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                }

                // Wyœwietlanie przetworzonego obrazu z oznaczonymi twarzami i informacj¹ o p³ci
                displayImage(image, "Obraz po przetworzeniu");
            }
            else {
                // Inkrementowanie licznika zdjêæ bez twarzy
                imagesWithoutFaces++;
                std::cout << "Brak wykrytych twarzy w obrazie: " << entry.path().filename() << std::endl;
            }
        }
    }

    // Wyœwietlanie wyników
    std::cout << "Liczba wykrytych twarzy: " << totalFaces << std::endl;
    std::cout << "Liczba zdjec bez wykrytych twarzy: " << imagesWithoutFaces << std::endl;

    std::cout << "Przetwarzanie zakonczone." << std::endl;

    return 0;
}
