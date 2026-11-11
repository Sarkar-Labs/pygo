package tensor

import "fmt"

func Fused1355SiluAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Sigmoid()
	return r, nil
}
func Fused1355SiluAddSigmoidName() string { return "silu_add_sigmoid" }
