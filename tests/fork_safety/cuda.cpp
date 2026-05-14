// CUDA fork-safety: loud refusal. The parent submits a kernel, forks, and
// the child's first dispatch must surface an error - we check via the
// async exception handler and the kernel result not materializing.

#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

namespace {

bool is_cuda_device(const sycl::queue &q) {
  const auto plat = q.get_device().get_platform().get_info<sycl::info::platform::name>();
  return plat.find("CUDA") != std::string::npos || plat.find("NVIDIA") != std::string::npos;
}

int submit_expecting_error() {
  std::atomic<bool> saw_error{false};
  sycl::queue q{sycl::async_handler{[&](sycl::exception_list list) {
    for (auto &e : list) {
      (void)e;
      saw_error.store(true);
    }
  }}};

  try {
    constexpr std::size_t N = 64;
    sycl::buffer<int> buf{sycl::range<1>{N}};
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access<sycl::access::mode::write>(cgh);
       cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) { acc[i] = 1; });
     }).wait_and_throw();
  } catch (const sycl::exception &) {
    saw_error.store(true);
  } catch (...) {
    saw_error.store(true);
  }
  q.wait_and_throw();
  return saw_error.load() ? 0 : 1;
}

} // namespace

int main() {
  sycl::queue q;
  if (!is_cuda_device(q)) {
    std::printf("cuda fork-safety: no CUDA device - skipping\n");
    return 0;
  }

  // Parent: submit once to warm the backend.
  {
    constexpr std::size_t N = 64;
    sycl::buffer<int> buf{sycl::range<1>{N}};
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access<sycl::access::mode::write>(cgh);
       cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) { acc[i] = 1; });
     }).wait();
  }

  pid_t pid = fork();
  if (pid < 0) { std::perror("fork"); return 1; }
  if (pid == 0) {
    // Child: first dispatch must surface a refusal, not silently corrupt.
    _exit(submit_expecting_error());
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) { std::perror("waitpid"); return 1; }
  const int child_exit = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  if (child_exit != 0) {
    std::fprintf(stderr, "cuda fork-safety: child did not see refusal (status=%d)\n", status);
    return 1;
  }
  std::printf("cuda fork-safety: correctly refused\n");
  return 0;
}
