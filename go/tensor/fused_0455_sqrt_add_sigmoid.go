package tensor

import "fmt"

func Fused0455SqrtAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.Sigmoid()
	return r, nil
}
func Fused0455SqrtAddSigmoidName() string { return "sqrt_add_sigmoid" }
