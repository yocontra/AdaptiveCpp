// HIP fork-safety: outcome depends on the HIP target.
//   HIP/ROCm  — libhsakmt supports fork-without-exec; child must succeed.
//   HIP/CUDA  — NVIDIA forbids fork-without-exec; child must see an error
//               and exit non-zero.

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

namespace {

int run_kernel(sycl::queue &q) {
  constexpr std::size_t N = 1024;
  std::vector<int> host(N, 0);
  try {
    sycl::buffer<int> buf{host.data(), sycl::range<1>{N}};
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access<sycl::access::mode::write>(cgh);
       cgh.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { acc[i] = static_cast<int>(i[0]); });
     }).wait();
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "hip fork-safety: submit threw: %s\n", e.what());
    return 1;
  }
  for (std::size_t i = 0; i < N; ++i) {
    if (host[i] != static_cast<int>(i)) return 1;
  }
  return 0;
}

bool is_hip_device(const sycl::queue &q) {
  const auto plat = q.get_device().get_platform().get_info<sycl::info::platform::name>();
  return plat.find("HIP") != std::string::npos || plat.find("hip") != std::string::npos ||
         plat.find("ROCm") != std::string::npos || plat.find("AMD") != std::string::npos;
}

} // namespace

int main() {
  sycl::queue q;
  if (!is_hip_device(q)) {
    std::printf("hip fork-safety: no HIP device — skipping\n");
    return 0;
  }

  if (run_kernel(q) != 0) return 1;

  pid_t pid = fork();
  if (pid < 0) { std::perror("fork"); return 1; }
  if (pid == 0) {
    sycl::queue child_q;
    _exit(run_kernel(child_q));
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) { std::perror("waitpid"); return 1; }
  const int child_exit = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

#if defined(HIPSYCL_RT_HIP_TARGET_CUDA)
  // HIP-over-CUDA must refuse.
  if (child_exit == 0) {
    std::fprintf(stderr, "hip fork-safety (over CUDA): child unexpectedly succeeded\n");
    return 1;
  }
  std::printf("hip fork-safety (over CUDA): correctly refused\n");
  return 0;
#else
  // HIP-over-ROCm / HIP-CPU: must transparently recover.
  if (child_exit != 0) {
    std::fprintf(stderr, "hip fork-safety: child failed (status=%d)\n", status);
    return 1;
  }
  std::printf("hip fork-safety: OK\n");
  return 0;
#endif
}
