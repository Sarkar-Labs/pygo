package tensor

import "fmt"

func Fused1536Log2SubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Sigmoid()
	return r, nil
}
func Fused1536Log2SubSigmoidName() string { return "log2_sub_sigmoid" }
