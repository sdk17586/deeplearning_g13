#include "file_utils.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>

std::vector<std::string> readLines(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open tensor list: " + path);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

std::vector<char> readBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open binary file: " + path);
  }
  return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}

void writeBinary(const std::string& path, const void* data, std::size_t size) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to write file: " + path);
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}
