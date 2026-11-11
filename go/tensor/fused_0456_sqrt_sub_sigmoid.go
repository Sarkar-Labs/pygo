package tensor

import "fmt"

func Fused0456SqrtSubSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_sigmoid: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Sigmoid()
	return r, nil
}
func Fused0456SqrtSubSigmoidName() string { return "sqrt_sub_sigmoid" }
