package tensor

import "fmt"

func Fused0815CosAddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Sigmoid()
	return r, nil
}
func Fused0815CosAddSigmoidName() string { return "cos_add_sigmoid" }
