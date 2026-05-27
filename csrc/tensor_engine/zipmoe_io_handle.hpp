#pragma once



static const size_t  kAioAlignment = 4096;


class ZipMoEIOHandle{

public:
    explicit ZipMoEIOHandle(const std::string& prefix);
    ~ZipMoEIOHandle();

    std::int64_t Read(const std::string& filename, void* buffer,
                        const bool high_prio, const int64_t num_bytes,
                        const std::int64_t offset);
    std::int64_t Write(const std::string& filename, const void* buffer,
                        const bool high_prio, const int64_t num_bytes,
                        const std::int64_t offset);

private:
    void Run();  // io submit thread function

    private:
    bool time_to_exit_;
    std::thread thread_;
    std::mutex file_set_mutex_;
    std::unordered_map<std::string, int> file_set_;
};