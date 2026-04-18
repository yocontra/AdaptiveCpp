// Metal fork-safety: transparent recovery. The parent submits a trivial
// kernel, forks, and the child must succeed on a fresh submission. The
// backend's internal pid_guard detects the fork at get_executor() entry,
// drops inherited MTL::* handles via kernel_cache / hw_manager reset, and
// the post-fork submission reloads the .metallib + .metalar from disk
// without touching MTLCompilerService.

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

namespace {

int run_kernel(sycl::queue &q) {
  constexpr std::size_t N = 1024;
  std::vector<int> host(N, 0);
  {
    sycl::buffer<int> buf{host.data(), sycl::range<1>{N}};
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access<sycl::access::mode::write>(cgh);
       cgh.parallel_for(sycl::range<1>{N},
                        [=](sycl::id<1> i) { acc[i] = static_cast<int>(i[0]); });
     }).wait();
  }
  for (std::size_t i = 0; i < N; ++i) {
    if (host[i] != static_cast<int>(i)) {
      std::fprintf(stderr, "metal fork-safety: bad result at %zu: %d\n", i, host[i]);
      return 1;
    }
  }
  return 0;
}

} // namespace

int main() {
  sycl::queue q;
  const auto plat = q.get_device().get_platform().get_info<sycl::info::platform::name>();
  if (plat.find("Metal") == std::string::npos && plat.find("metal") == std::string::npos) {
    std::printf("metal fork-safety: no Metal device — skipping\n");
    return 0;
  }

  if (run_kernel(q) != 0) return 1;

  pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork");
    return 1;
  }
  if (pid == 0) {
    // Child — first dispatch must trigger the reset path and succeed.
    sycl::queue child_q;
    _exit(run_kernel(child_q));
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::perror("waitpid");
    return 1;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "metal fork-safety: child failed (status=%d)\n", status);
    return 1;
  }
  std::printf("metal fork-safety: OK\n");
  return 0;
}
