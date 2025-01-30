#ifndef testlib
#define testlib

#include <cstdlib>
#include <chrono>
#include <format>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <signal.h>
#include <string>
#include <thread>

#include <cassert>

#define INDEX(N,R,C) (N*R + C)
#ifndef MATRIX_N
#define MATRIX_N (-1)
#endif

constexpr unsigned int APPLE_M1_PAGE_SIZE = 16384;

float* internal_make_page_aligned_matrix(unsigned int n, unsigned int &memory_length) {
    unsigned int required_size = n * n * sizeof(float);
    unsigned int rem = required_size % APPLE_M1_PAGE_SIZE;
    memory_length = rem == 0 ? required_size : required_size + APPLE_M1_PAGE_SIZE - rem;
    auto* arr = (float*) aligned_alloc(APPLE_M1_PAGE_SIZE, memory_length);
    assert(arr != nullptr);
    return arr;
}

void internal_load_matrix(const std::string& pathPrefix, unsigned short n, unsigned short idx, float* arr) {
    auto file_name = pathPrefix + "matrix-" + std::to_string(n) + "-" + std::to_string(idx) + ".float32";
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "could not open '" << file_name << "'" << std::endl;
        return;
    }
    if (!file.read(reinterpret_cast<char*>(arr), static_cast<std::streamsize>(n * n * sizeof(float)))) {
        // Successfully read a float
        std::cerr << "error reading '" << file_name << "' as float32" << std::endl;
        return;
    }
}

void internal_power_sample() {
    const char* pid_string = std::getenv("POWER_MONITOR");
    if (pid_string == nullptr) {
        std::cerr << "POWER_MONITOR not set, monitor end notification not sent" << std::endl;
        return;
    }
    char* end;
    int pid = static_cast<int>(std::strtol(pid_string, &end, 10));
    if (*end != '\0') {
        std::cerr << "POWER_MONITOR is not a valid integer" << std::endl;
        return;
    }
    kill(pid, SIGINFO);
}

template<typename MatMul>
void test_suite(MatMul consumer, const std::string &pathPrefix = "../data/") {
    unsigned int n = MATRIX_N;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "Simplify"
    if (n == -1) {
        std::cerr << "Matrix size not set up properly at compile time. Using default of 150." << std::endl;
        n = 150u;
    }
#pragma clang diagnostic pop
    // Create all the arrays.
    unsigned int memory_length;
    float* left = internal_make_page_aligned_matrix(n, memory_length);
    float* right = internal_make_page_aligned_matrix(n, memory_length);
    float* out = internal_make_page_aligned_matrix(n, memory_length);
    // Load them.
    internal_load_matrix(pathPrefix, n, 0, left);
    internal_load_matrix(pathPrefix, n, 1, right);
    // Perform matrix multiplication.
    internal_power_sample();
    auto before = std::chrono::high_resolution_clock::now();
    consumer(n, memory_length, left, right, out);
    auto after = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(after - before);

    // powermetrics sample periods should be longer than 10ms, otherwise it ignores the second signal
    if (elapsed.count() < 10000000) {
        // sleep to ensure the second signal is received
        std::this_thread::sleep_for(std::chrono::milliseconds(10 - elapsed.count() / 1000000));
    }
    internal_power_sample(); 
#ifndef NDEBUG
    std::cout << n << ": check " << out[INDEX(n, 0, 0)] << " took " << elapsed.count() << "ns" << std::endl;
#else
    std::cout << elapsed.count() << std::endl;
#endif
}

#endif
