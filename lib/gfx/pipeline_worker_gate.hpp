#pragma once

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace aurora::gfx {

// Serializes a narrow external initialization window against pipeline creation. A pause first
// closes the gate to new work, then waits for every worker that already crossed it to leave.
class PipelineWorkerGate {
public:
  class WorkLease {
  public:
    WorkLease(const WorkLease&) = delete;
    WorkLease& operator=(const WorkLease&) = delete;
    WorkLease(WorkLease&& other) noexcept : m_gate(other.m_gate) { other.m_gate = nullptr; }
    WorkLease& operator=(WorkLease&&) = delete;

    ~WorkLease() {
      if (m_gate != nullptr) {
        m_gate->leave_work();
      }
    }

  private:
    friend class PipelineWorkerGate;
    explicit WorkLease(PipelineWorkerGate& gate) : m_gate(&gate) {}

    PipelineWorkerGate* m_gate;
  };

  [[nodiscard]] WorkLease enter_work() {
    std::unique_lock lock{m_mutex};
    m_stateChanged.wait(lock, [this] { return !m_paused; });
    ++m_activeWorkers;
    return WorkLease{*this};
  }

  void pause_and_wait() {
    std::unique_lock lock{m_mutex};
    m_paused = true;
    m_stateChanged.notify_all();
    m_stateChanged.wait(lock, [this] { return m_activeWorkers == 0; });
  }

  void resume() {
    {
      std::lock_guard lock{m_mutex};
      m_paused = false;
    }
    m_stateChanged.notify_all();
  }

  [[nodiscard]] bool is_paused() const {
    std::lock_guard lock{m_mutex};
    return m_paused;
  }

private:
  void leave_work() {
    {
      std::lock_guard lock{m_mutex};
      assert(m_activeWorkers > 0);
      --m_activeWorkers;
    }
    m_stateChanged.notify_all();
  }

  mutable std::mutex m_mutex;
  std::condition_variable m_stateChanged;
  size_t m_activeWorkers = 0;
  bool m_paused = false;
};

} // namespace aurora::gfx
