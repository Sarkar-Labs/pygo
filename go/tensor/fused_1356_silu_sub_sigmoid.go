package tensor

import "fmt"

func Fused1356SiluSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.SiLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1356SiluSubSigmoidName() string { return "silu_sub_sigmoid" }
