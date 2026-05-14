// Metal fork-safety: loud refusal. macOS Metal cannot safely allocate driver
// resources in a child forked from a process that has already used Metal. The
// child must see a controlled AdaptiveCpp error instead of crashing in the
// Metal driver.

#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
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

int submit_expecting_error() {
  std::atomic<bool> saw_error{false};
  sycl::queue q{sycl::async_handler{[&](sycl::exception_list list) {
    for (auto &e : list) {
      (void)e;
      saw_error.store(true);
    }
  }}};

  int rc = 1;
  try {
    rc = run_kernel(q);
  } catch (const sycl::exception &) {
    saw_error.store(true);
  } catch (...) {
    saw_error.store(true);
  }
  try {
    q.wait_and_throw();
  } catch (const sycl::exception &) {
    saw_error.store(true);
  } catch (...) {
    saw_error.store(true);
  }
  return saw_error.load() || rc != 0 ? 0 : 1;
}

} // namespace

int main() {
  sycl::queue q;
  const auto plat = q.get_device().get_platform().get_info<sycl::info::platform::name>();
  if (plat.find("Metal") == std::string::npos && plat.find("metal") == std::string::npos) {
    std::printf("metal fork-safety: no Metal device - skipping\n");
    return 0;
  }

  if (run_kernel(q) != 0) return 1;

  pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork");
    return 1;
  }
  if (pid == 0) {
    // Child: first dispatch must surface a refusal, not crash in Metal.
    _exit(submit_expecting_error());
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::perror("waitpid");
    return 1;
  }
  const int child_exit = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  if (child_exit != 0) {
    std::fprintf(stderr,
                 "metal fork-safety: child did not see refusal (status=%d)\n",
                 status);
    return 1;
  }
  std::printf("metal fork-safety: correctly refused\n");
  return 0;
}
