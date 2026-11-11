package tensor

import "fmt"

func Fused1506NegSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("neg_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Neg()
	r = r.GELU()
	return r, nil
}
func Fused1506NegSubGeluName() string { return "neg_sub_gelu" }
