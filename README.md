# Music Genre Classifier

## Overview
The Music Genre Classifier project focuses on classifying music tracks into different genres using a combination of Short-Time Fourier Transform (STFT), implemented with a custom Fast Fourier Transform (FFT) function, five different windowing techniques, and three different K-nearest neighbors (KNN) k values.

## Project Details
The main objective of this project is to develop a music genre classifier that can accurately classify music tracks into various genres. The following key components and techniques are employed:

- **Custom Multi Threaded FFT Implementation**: The project utilizes a custom FFT function to extract frequency information from audio signals, forming the basis for genre classification.
- **Multi Thread Short-Time Fourier Transform (STFT)**: The audio signal is divided into overlapping frames, and the FFT is applied to each frame. This generates a time-frequency representation known as the spectrogram, which captures the frequency content of the audio signal.
- **Windowing Techniques**: Five different windowing techniques (e.g., Hamming, Hanning, Blackman) are used to improve the accuracy of the spectrogram representation by reducing spectral leakage.
- **K-Nearest Neighbors (KNN) Classification**: The extracted spectrogram features are used to train a KNN classifier with three different values for the K parameter. This enables the classification of music tracks into predefined genres.

## Key Features
1. **Custom FFT Implementation**: The project showcases an efficient and accurate custom FFT function, allowing for precise frequency analysis and feature extraction.
2. **Genre Classification**: The classifier accurately predicts the genre of music tracks based on their spectrogram features, enabling automated genre labeling.
3. **Flexible Windowing Techniques**: Five different windowing techniques are implemented, providing users with the ability to experiment and choose the most suitable method for their specific genre classification tasks.
4. **KNN Classification with Multiple K Values**: The project tests the KNN classifier with three different K values, enabling the evaluation and comparison of classification performance.
5. **Performance Statistics and Evaluation**: The project collects statistics such as accuracy, precision, recall, and F1 score for each combination of windowing technique and K value, providing insights into the performance of the music genre classifier.
6. **Example Code and Usage**: Detailed usage instructions, along with code snippets, are included to guide users in utilizing the music genre classification system with their own audio tracks.

## Dataset
To train and evaluate the music genre classifier, a labeled dataset of audio tracks with genre annotations is required. You can use publicly available music datasets or create your own dataset by collecting and labeling music tracks for different genres.

## Contributions and Feedback
Contributions and feedback to improve the Music Genre Classifier project are highly appreciated. If you encounter any issues, have suggestions for enhancements, or would like to contribute improvements, please submit an issue or pull request on the GitHub repository.

## License
This project is licensed under the [MIT License](LICENSE), allowing you to use, modify, and distribute the codebase as permitted by the license.

**Disclaimer:** The Music Genre Classifier project is provided as-is, and the authors assume no liability for any damages or misuse of the software. Users are responsible for ensuring compliance with applicable laws and regulations.

---
