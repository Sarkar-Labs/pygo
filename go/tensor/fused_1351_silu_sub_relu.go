package tensor

import "fmt"

func Fused1351SiluSubRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_relu: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.ReLU()
	return r, nil
}
func Fused1351SiluSubReluName() string { return "silu_sub_relu" }
