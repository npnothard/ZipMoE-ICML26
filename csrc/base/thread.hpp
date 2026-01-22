// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)


#ifndef MUDUO_BASE_THREAD_H
#define MUDUO_BASE_THREAD_H

#include "logs/countdown_latch.hpp"
#include "logs/types.hpp"

#include <pthread.h>
#include <atomic>
#include <functional>
#include <memory>
#include <string>

namespace base {

class Thread : noncopyable {
 public:
  typedef std::function<void()> ThreadFunc;

  explicit Thread(ThreadFunc, const std::string& name = std::string());
  ~Thread();

  void start();
  int join();

  bool started() const { return started_; }
  pid_t tid() const { return tid_; }
  const std::string& name() const { return name_; }

  static int numCreated() { return numCreated_.load(); }

 private:
  void setDefaultName();

  bool started_;
  bool joined_;
  pthread_t pthreadId_;
  pid_t tid_;
  ThreadFunc func_;
  std::string name_;
  CountDownLatch latch_;

  static std::atomic_int32_t numCreated_;
};

}
#endif
