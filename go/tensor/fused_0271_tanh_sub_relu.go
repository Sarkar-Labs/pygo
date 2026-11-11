package tensor

import "fmt"

func Fused0271TanhSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.ReLU()
	return r, nil
}
func Fused0271TanhSubReluName() string { return "tanh_sub_relu" }
