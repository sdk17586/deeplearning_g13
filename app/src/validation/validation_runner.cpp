#include "validation_runner.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <gst/gst.h>

#include "action_recognition_pipeline.hpp"

namespace {

struct ValidationSample {
  std::filesystem::path video_path;
  std::string expected_label;
};

constexpr std::array<const char *, 5> kKnownLabels = {
    "abandon", "fight", "broken", "theft", "normal"};

bool IsKnownLabel(const std::string &label) {
  return std::find(kKnownLabels.begin(), kKnownLabels.end(), label) != kKnownLabels.end();
}

std::vector<ValidationSample> LoadSamples(const std::filesystem::path &dataset_dir) {
  std::vector<ValidationSample> samples;
  const std::filesystem::path clips_dir = dataset_dir / "clips";

  if (!std::filesystem::is_directory(clips_dir)) {
    g_printerr("[검증 오류] clips 디렉터리가 없습니다: %s\n", clips_dir.c_str());
    return samples;
  }

  for (const auto &class_entry : std::filesystem::directory_iterator(clips_dir)) {
    if (!class_entry.is_directory()) {
      continue;
    }

    const std::string expected_label = class_entry.path().filename().string();
    if (!IsKnownLabel(expected_label)) {
      g_printerr("[검증 경고] 알 수 없는 클래스 디렉터리는 건너뜁니다: %s\n",
                 class_entry.path().c_str());
      continue;
    }

    for (const auto &video_entry : std::filesystem::directory_iterator(class_entry.path())) {
      if (!video_entry.is_regular_file()) {
        continue;
      }
      samples.push_back({video_entry.path(), expected_label});
    }
  }

  std::sort(samples.begin(), samples.end(),
            [](const ValidationSample &lhs, const ValidationSample &rhs) {
              return lhs.video_path.string() < rhs.video_path.string();
            });
  return samples;
}

void CheckLabelDirectories(const std::filesystem::path &dataset_dir) {
  const std::filesystem::path labels_dir = dataset_dir / "labels";
  if (!std::filesystem::is_directory(labels_dir)) {
    g_printerr("[검증 경고] labels 디렉터리가 없습니다: %s\n", labels_dir.c_str());
    return;
  }

  for (const char *label : kKnownLabels) {
    const std::filesystem::path label_dir = labels_dir / label;
    if (std::string(label) == "normal" && !std::filesystem::exists(label_dir)) {
      g_print("[검증 안내] normal 클래스는 labels/normal 없이 clips/normal을 정답으로 사용합니다.\n");
      continue;
    }

    if (!std::filesystem::is_directory(label_dir)) {
      g_printerr("[검증 경고] labels/%s 디렉터리가 없습니다.\n", label);
      continue;
    }

    size_t xml_count = 0;
    for (const auto &entry : std::filesystem::directory_iterator(label_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".xml") {
        ++xml_count;
      }
    }
    if (xml_count == 0) {
      g_printerr("[검증 경고] labels/%s 디렉터리에 XML 파일이 없습니다.\n", label);
    }
  }
}

}  // namespace

int RunValidation(const AppConfig &config) {
  const std::filesystem::path dataset_dir = config.validation_dir;
  CheckLabelDirectories(dataset_dir);

  const std::vector<ValidationSample> samples = LoadSamples(dataset_dir);
  if (samples.empty()) {
    g_printerr("[검증 오류] 검증할 영상이 없습니다: %s\n", dataset_dir.c_str());
    return EXIT_FAILURE;
  }

  g_print("\n========== 검증 시작 ==========\n");
  g_print("검증 데이터셋: %s\n", dataset_dir.c_str());
  g_print("총 영상 수: %zu\n", samples.size());
  g_print("추론 설정: %s\n", config.infer_config.c_str());
  g_print("전처리 설정: %s\n", config.preprocess_config.c_str());
  g_print("================================\n\n");

  size_t correct = 0;
  size_t failed = 0;

  for (size_t i = 0; i < samples.size(); ++i) {
    const ValidationSample &sample = samples[i];
    AppConfig sample_config = config;
    sample_config.mode = AppConfig::Mode::PredictOne;
    sample_config.uri = AppConfig::ToUri(sample.video_path.c_str());
    sample_config.print_predictions = false;
    sample_config.print_summary = false;

    ActionRecognitionPipeline pipeline(sample_config);
    const int status = pipeline.Run();
    const auto predicted_label = pipeline.PredictedLabelId();

    bool is_correct = false;
    if (status == EXIT_SUCCESS && predicted_label.has_value()) {
      is_correct = predicted_label.value() == sample.expected_label;
      if (is_correct) {
        ++correct;
      }
    } else {
      ++failed;
    }

    g_print("[%zu/%zu] %s  정답=%s  예측=%s  %s\n",
            i + 1, samples.size(), sample.video_path.filename().c_str(),
            sample.expected_label.c_str(),
            predicted_label.has_value() ? predicted_label->c_str() : "none",
            is_correct ? "OK" : "MISS");
  }

  const double accuracy = samples.empty()
                              ? 0.0
                              : static_cast<double>(correct) * 100.0 /
                                    static_cast<double>(samples.size());

  g_print("\n========== 검증 결과 ==========\n");
  g_print("정답 수: %zu/%zu\n", correct, samples.size());
  g_print("실패/예측없음: %zu\n", failed);
  g_print("Accuracy: %.2f%%\n", accuracy);
  g_print("================================\n");

  return failed == samples.size() ? EXIT_FAILURE : EXIT_SUCCESS;
}
