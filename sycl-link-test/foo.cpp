#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl;

int main(int argc, char **argv){

  const int N = 1023 + argc;
  
  sycl::device device{};
  sycl::queue queue{device};
  double * d_a = sycl::malloc_device<double>(N, queue);
  double * d_b = sycl::malloc_device<double>(1, queue);
  
  std::vector<double> h_a(N);
  for(int ix=0 ; ix<N ; ix++){
    h_a[ix] = (double) ix;
  }
  queue.memcpy(d_a, h_a.data(), N*sizeof(double)).wait_and_throw();
  
  queue.submit([&] (sycl::handler& h) {
    sycl::accessor<
      double, 
      1, 
      sycl::access::mode::read_write, 
      sycl::access::target::local
    > local_mem(sycl::range<1>(32), h);

    h.parallel_for(
      sycl::nd_range<1>(
        sycl::range<1>(N),
        sycl::range<1>(32)
      ),
      [=] (sycl::nd_item<1> id) {
        const size_t lid = id.get_local_linear_id();
        const size_t gid = id.get_global_linear_id();
        local_mem[lid] = d_a[gid];
        id.barrier(sycl::access::fence_space::local_space);
        d_b[gid] = local_mem[(lid + 1) % 32];
      });
  }).wait_and_throw();

  queue.memcpy(h_a.data(), d_b, N*sizeof(double)).wait_and_throw();
 
  for(int ix=0 ; ix<N ; ix++){
    std::cout << h_a[ix] << std::endl;
  }

  sycl::free(d_a, queue);
  sycl::free(d_b, queue);

  return 0;
}
