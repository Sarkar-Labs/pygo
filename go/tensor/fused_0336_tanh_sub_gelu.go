package tensor

import "fmt"

func Fused0336TanhSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.GELU()
	return r, nil
}
func Fused0336TanhSubGeluName() string { return "tanh_sub_gelu" }
