// Copyright (c) 2026 <ZipMoE / Anonymous Team>.
// All rights reserved.
//
// This source code is licensed under the Academic Non-Commercial License.
// See the LICENSE file in the project root for details.


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include "dispatcher/expert_dispatcher.hpp"
#include "prefetch/zipmoe_prefetch_handle.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ZipMoE, m){
    py::class_<ZipMoEPrefetchHandle>(m, "zipmoe_prefetch_handle")
        .def(
            py::init<
                const std::string&,
                const std::string&,
                const std::string&,
                const std::string&,
                const double&,
                const double&,
                double,
                double,
                int,
                int,
                int,
                int,
                size_t,
                size_t,
                int,
                int,
                int,
                int,
                int,
                double,
                bool
            >(),
            py::arg("offload_folder"),
            py::arg("offload_file_name"),
            py::arg("CODE_TYPE"),
            py::arg("caching_algorithm"),
            py::arg("device_memory_ratio"),
            py::arg("gpu_pool_ratio"),
            py::arg("decompression_delay"),
            py::arg("sm_io_delay"),
            py::arg("num_compute_threads"),
            py::arg("num_file_chunks"),
            py::arg("prefetcher_topk"),
            py::arg("expert_topk"),
            py::arg("num_elements_per_expert"),
            py::arg("num_tensors_per_expert"),
            py::arg("num_expert_layers"),
            py::arg("num_experts"),
            py::arg("LZ4_accelerationLevel"),
            py::arg("LZ4HC_compressionLevel"),
            py::arg("ZSTD_compressionLevel"),
            py::arg("hyperparam_state_margin"),
            py::arg("bind_core")
        )

        .def(
            "open_offload_file",
            &ZipMoEPrefetchHandle::OpenOffloadFile,
            "Open the offloaded file after the offloading is completed."
        )

        .def(
            "offload",
            &ZipMoEPrefetchHandle::OffloadTensor,
            "Offload a tensor."
        )

        .def(
            "batch_offload",
            &ZipMoEPrefetchHandle::BatchOffloadTensor,
            "Offload a tensor in a batch."
        )

        .def(
            "register",
            &ZipMoEPrefetchHandle::RegisterTensor,
            "register a buffer into the C++ engine hash table."
        )

        .def(
            "begin",
            &ZipMoEPrefetchHandle::AcquireTensor,
            "Operates as a pre hook, registered for all nodes."
        )


        .def(
            "end",
            &ZipMoEPrefetchHandle::ReleaseTensor,
            "Operates as a post hook, registered for all nodes."
        )


        .def(
            "set_topology",
            &ZipMoEPrefetchHandle::SetTopology,
            "Set up model topology and move all dense nodes to GPU."
        )

        .def(
            "update_tensor_map",
            &ZipMoEPrefetchHandle::UpdateTensorMap,
            "Used in cast_classifier_decorator for patching the SwitchTransformersTop1Router._cast_classifier."
        )        

        .def(
            "is_tensor_offloaded",
            &ZipMoEPrefetchHandle::IsTensorOffloaded,
            "Check whether a tensor is offloaded."
        )

        .def(
            "is_tensor_index_initialized",
            &ZipMoEPrefetchHandle::IsTensorIndexInitialized,
            "Check whether all tensors have been offloaded. (So that we have proper tensor index)."
        )

        .def(
            "get_node_default_device",
            &ZipMoEPrefetchHandle::GetNodeDefaultDevice,
            "Returns the device index of a given node."
        )

        .def(
            "fetch_tensors",
            &ZipMoEPrefetchHandle::FetchTensors,
            "Operates as a pre hook, registered for all non-expert nodes."
        )

        .def(
            "fetch_tensors",
            &ZipMoEPrefetchHandle::FetchTensors,
            "Operates as a pre hook, registered for all non-expert nodes."
        )

        .def(
            "reset_access_counts",
            &ZipMoEPrefetchHandle::ResetAccessCounts,
            "Reset the global_access_counts to prevent inter batch interference."
        );


    py::class_<ExpertDispatcher>(m, "expert_dispatcher")
        .def(
            py::init<
                int,
                int,
                int,
                int
            >(),
            py::arg("num_experts"),
            py::arg("num_layers"),
            py::arg("expert_type"),
            py::arg("dtype")
        )

        .def(
            "register_expert", 
            &ExpertDispatcher::RegisterExpert,
            "Register the experts."
        )

        .def(
            "obtain_tensor", 
            &ExpertDispatcher::ObtainTensor,
            "Obtain the tensor for debugging."
        )

        .def(
            "enqueue_expert", 
            &ExpertDispatcher::EnqueueExpert,
            "Push the on-demand fetch request into the new_task_queue in TensorEngine."
        )


        .def(
            "enqueue_prefetch",
            &ExpertDispatcher::EnqueuePrefetch,
            "Push the prefetch request into the new_task_queue in TensorEngine."
        )

        .def(
            "submit_prefetch",
            &ExpertDispatcher::SubmitPrefetch,
            "Submit the task into workers. Push the prefetch request into the scheduled_queue in TensorEngine."
        )


        .def(
            "ops_schedule",
            &ExpertDispatcher::OperationSchedule,
            "Call the TensorEngine scheduler to schedule the operations."
        )

        .def(
            "set_inputs",
            &ExpertDispatcher::SetInputs,
            "Clone the hidden_states and the router mask before forward of the sparse layer."
        )

        .def(
            "set_expected_queue",
            &ExpertDispatcher::SetExpectedQueue,
            "Reister the queue length for synchronization."
        )

        .def(
            "wait_expert",
            &ExpertDispatcher::WaitExpert,
            py::call_guard<py::gil_scoped_release>(),
            "Wait for all the experts to complete and collect the result."
        );

}