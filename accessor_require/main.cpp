#include <CL/sycl.hpp>
using namespace cl;

#include <type_traits>
#include <vector>
#include <cstdlib>
#include <memory>
#include <list>
#include <iostream>
#include <vector>

struct DeviceType {
  sycl::accessor<double, 1, sycl::access_mode::read> a_data;
  inline double get(const int idx) const {
    return a_data[idx];
  }
};

struct HostType {

  std::vector<double> data;
  sycl::buffer<double, 1> b_data;

  HostType(
    const int N
  ) : b_data(0) {
    data = std::vector<double>(N);
    for(int ix=0 ; ix<N ; ix++){
      data.at(ix) = ix;
    }
    this->b_data = sycl::buffer<double, 1>(this->data.data(), N);
  }

  inline DeviceType get_device(sycl::handler &cgh){
    DeviceType dt = {
      this->b_data.get_access<sycl::access_mode::read>(cgh)
    };
    return dt;
  }

};

int main(int argc, char** argv){

  sycl::device device = sycl::device(sycl::default_selector());
  std::cout << "Using " << device.get_info<sycl::info::device::name>()
    << std::endl;
  sycl::queue queue = sycl::queue(device);

  const std::size_t N = 1024;

  std::vector<double> h_output(N);
  sycl::buffer<double, 1>  b_output(h_output.data(), N);

  HostType host_type(N);

  queue.submit([&](sycl::handler &cgh) {
      auto a_output = b_output.get_access<sycl::access_mode::read_write>(cgh);
      DeviceType device_type = host_type.get_device(cgh);
      cgh.parallel_for<>(
          sycl::range<1>(N), [=](sycl::id<1> idx) {
            a_output[idx] = device_type.get(idx);
          });
    })
    .wait_and_throw();
  
  auto a_output = b_output.get_access<sycl::access_mode::read>();
  std::cout << a_output[0] << std::endl;
  std::cout << a_output[N-1] << std::endl;
  return 0;
}

