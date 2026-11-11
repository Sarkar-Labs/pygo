package tensor

import "fmt"

func Fused1326GeluSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.GELU()
	return r, nil
}
func Fused1326GeluSubGeluName() string { return "gelu_sub_gelu" }
