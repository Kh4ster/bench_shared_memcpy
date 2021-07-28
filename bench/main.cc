#include "cuda_tools/host_shared_ptr.cuh"
#include "test_helpers.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

class Fixture : public benchmark::Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void
    bench(benchmark::State& st, FUNC callback, std::size_t size, Args&&... args)
    {
        cuda_tools::host_shared_ptr<int> result(size);
        cuda_tools::host_shared_ptr<int> global1(size);
        cuda_tools::host_shared_ptr<int> global2(size);

        for (auto _ : st)
            callback(result, global1, global2, std::forward<Args>(args)...);

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        // if (!no_check)
        //    check_buffer(buffer);
    }
};

bool Fixture::no_check = false;

BENCHMARK_DEFINE_F(Fixture, Basic)
(benchmark::State& st) { this->bench(st, basic, std::size_t(1) << 26); }

BENCHMARK_DEFINE_F(Fixture, Cooperative_basic)
(benchmark::State& st)
{
    this->bench(st, cooperative_basic, std::size_t(1) << 26);
}

BENCHMARK_REGISTER_F(Fixture, Basic)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(Fixture, Cooperative_basic)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);

    for (int i = 1; i < argc; i++)
    {
        if (argv[i] == std::string_view("--no-check"))
        {
            Fixture::no_check = true;
            std::swap(argv[i], argv[--argc]);
        }
    }

    ::benchmark::RunSpecifiedBenchmarks();
}
