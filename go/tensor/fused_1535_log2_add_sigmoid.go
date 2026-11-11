package tensor

import "fmt"

func Fused1535Log2AddSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_sigmoid: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Sigmoid()
	return r, nil
}
func Fused1535Log2AddSigmoidName() string { return "log2_add_sigmoid" }
