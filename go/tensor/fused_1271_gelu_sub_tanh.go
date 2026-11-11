package tensor

import "fmt"

func Fused1271GeluSubTanh(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_tanh: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Tanh()
	return r, nil
}
func Fused1271GeluSubTanhName() string { return "gelu_sub_tanh" }
