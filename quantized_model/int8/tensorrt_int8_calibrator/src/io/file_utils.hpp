#pragma once

#include <cstddef>
#include <string>
#include <vector>

std::vector<std::string> readLines(const std::string& path);
std::vector<char> readBinary(const std::string& path);
void writeBinary(const std::string& path, const void* data, std::size_t size);
