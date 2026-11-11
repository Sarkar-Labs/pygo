package tensor

import "fmt"

func Fused0116ReluSubExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_exp: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Exp()
	return r, nil
}
func Fused0116ReluSubExpName() string { return "relu_sub_exp" }
