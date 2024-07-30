#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define MAX_ARTICLES 1000
#define MAX_TEXT_LENGTH 1000
#define MAX_FEATURES 1000
#define LEARNING_RATE 0.01
#define EPOCHS 1000

typedef struct {
    char text[MAX_TEXT_LENGTH];
    int label;
} Article;

Article dataset[MAX_ARTICLES];
int article_count = 0;
char vocabulary[MAX_FEATURES][50];
int vocab_size = 0;
double weights[MAX_FEATURES];
double bias = 0.0;

void preprocess_text(char *text) {
    for (int i = 0; text[i]; i++) {
        text[i] = tolower(text[i]);
        if (!isalnum(text[i]) && text[i] != ' ') {
            text[i] = ' ';
        }
    }
}

int is_in_vocabulary(const char *word) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(vocabulary[i], word) == 0) {
            return i;
        }
    }
    return -1;
}

void add_to_vocabulary(const char *word) {
    if (is_in_vocabulary(word) == -1) {
        strcpy(vocabulary[vocab_size], word);
        vocab_size++;
    }
}

void extract_features(Article *article, int *features) {
    char text[MAX_TEXT_LENGTH];
    strcpy(text, article->text);
    preprocess_text(text);
    char *token = strtok(text, " ");
    while (token != NULL) {
        int index = is_in_vocabulary(token);
        if (index != -1) {
            features[index]++;
        } else {
            add_to_vocabulary(token);
            features[vocab_size - 1]++;
        }
        token = strtok(NULL, " ");
    }
}

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

void train_model() {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0.0;
        for (int i = 0; i < article_count; i++) {
            int features[MAX_FEATURES] = {0};
            extract_features(&dataset[i], features);
            double linear_model = bias;
            for (int j = 0; j < vocab_size; j++) {
                linear_model += weights[j] * features[j];
            }
            double prediction = sigmoid(linear_model);
            double error = dataset[i].label - prediction;
            bias += LEARNING_RATE * error;
            for (int j = 0; j < vocab_size; j++) {
                weights[j] += LEARNING_RATE * error * features[j];
            }
            total_error += fabs(error);
        }
    }
}

void evaluate_model() {
    for (int i = 0; i < article_count; i++) {
        int features[MAX_FEATURES] = {0};
        extract_features(&dataset[i], features);
        double linear_model = bias;
        for (int j = 0; j < vocab_size; j++) {
            linear_model += weights[j] * features[j];
        }
        double prediction = sigmoid(linear_model);
        int predicted_label = prediction >= 0.5 ? 1 : 0;
        printf("News: \"%s\"\n", dataset[i].text);
        printf("Result: %s\n", predicted_label ? "Fake" : "Genuine");
    }
}

void create_default_dataset(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error creating file");
        exit(EXIT_FAILURE);
    }
    const char *data[] = {
        "\"Breaking news: Celebrity caught in scandalous act\",1",
        "\"Scientists discover new species in the Amazon rainforest\",0",
        "\"Local man wins lottery for the second time\",0",
        "\"Fake news article claiming moon landing was a hoax\",1",
        "\"Researchers develop new treatment for cancer\",0",
        "\"Fake news: Government plans to ban coffee\",1",
        "\"Technology giant releases innovative new smartphone\",0",
        "\"Fake report on health benefits of chocolate goes viral\",1",
        "\"New study shows exercise reduces risk of heart disease\",0",
        "\"Hoax news story about aliens landing in rural town\",1",
        "\"Government announces new policies to boost economy\",0",
        "\"Fake news: Water turning frogs gay\",1",
        "\"Scientists find evidence of life on Mars\",0",
        "\"Fake news: Vaccines cause autism\",1",
        "\"Local hero saves child from burning building\",0",
        "\"Fake report on benefits of drinking bleach\",1",
        "\"Research shows positive impact of meditation on mental health\",0",
        "\"Fake news: Earth is flat\",1",
        "\"Breakthrough in renewable energy technology\",0",
        "\"Fake news: 5G causes COVID-19\",1"
    };
    for (int i = 0; i < 20; i++) {
        fprintf(file, "%s\n", data[i]);
    }
    fclose(file);
}

void read_dataset(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    char line[MAX_TEXT_LENGTH + 10];
    while (fgets(line, sizeof(line), file) && article_count < MAX_ARTICLES) {
        char *text_part = strtok(line, "\"");
        char *label_part = strtok(NULL, ",");
        if (text_part && label_part) {
            strcpy(dataset[article_count].text, text_part);
            dataset[article_count].label = atoi(label_part);
            article_count++;
        }
    }
    fclose(file);
}

int main() {
    const char *dataset_filename = "news_dataset.csv";
    FILE *file = fopen(dataset_filename, "r");
    if (!file) {
        create_default_dataset(dataset_filename);
    } else {
        fclose(file);
    }
    read_dataset(dataset_filename);
    for (int i = 0; i < MAX_FEATURES; i++) {
        weights[i] = 0.0;
    }
    train_model();
    evaluate_model();
    return 0;
}
