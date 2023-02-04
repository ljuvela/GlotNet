// https://github.com/pytorch/audio/blob/e062110b286ba0673d773588d70a8d41994086f2/torchaudio/csrc/lfilter.cpp#L99

class DifferentiableIIR : public torch::autograd::Function<DifferentiableIIR> {
 public:
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& waveform,
      const torch::Tensor& a_coeffs_normalized) {
    auto device = waveform.device();
    auto dtype = waveform.dtype();
    int64_t n_batch = waveform.size(0);
    int64_t n_channel = waveform.size(1);
    int64_t n_sample = waveform.size(2);
    int64_t n_order = a_coeffs_normalized.size(1);
    int64_t n_sample_padded = n_sample + n_order - 1;

    auto a_coeff_flipped = a_coeffs_normalized.flip(1).contiguous();

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto padded_output_waveform =
        torch::zeros({n_batch, n_channel, n_sample_padded}, options);

    if (device.is_cpu()) {
      cpu_lfilter_core_loop(waveform, a_coeff_flipped, padded_output_waveform);
    } else {
      lfilter_core_generic_loop(
          waveform, a_coeff_flipped, padded_output_waveform);
    }

    auto output = padded_output_waveform.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(n_order - 1, torch::indexing::None)});

    ctx->save_for_backward({waveform, a_coeffs_normalized, output});
    return output;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto a_coeffs_normalized = saved[1];
    auto y = saved[2];

    int64_t n_batch = x.size(0);
    int64_t n_channel = x.size(1);
    int64_t n_order = a_coeffs_normalized.size(1);

    auto dx = torch::Tensor();
    auto da = torch::Tensor();
    auto dy = grad_outputs[0];

    namespace F = torch::nn::functional;

    if (a_coeffs_normalized.requires_grad()) {
      auto dyda = F::pad(
          DifferentiableIIR::apply(-y, a_coeffs_normalized),
          F::PadFuncOptions({n_order - 1, 0}));

      da = F::conv1d(
               dyda.view({1, n_batch * n_channel, -1}),
               dy.view({n_batch * n_channel, 1, -1}),
               F::Conv1dFuncOptions().groups(n_batch * n_channel))
               .view({n_batch, n_channel, -1})
               .sum(0)
               .flip(1); // flip coeffs
    }

    if (x.requires_grad()) {
      dx = DifferentiableIIR::apply(dy.flip(2), a_coeffs_normalized).flip(2);
    }

    return {dx, da};
  }
};