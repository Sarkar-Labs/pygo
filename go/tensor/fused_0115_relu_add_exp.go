package tensor

import "fmt"

func Fused0115ReluAddExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_add_exp: nil operand")
	}
	r := x.Add(y)
	r = r.ReLU()
	r = r.Exp()
	return r, nil
}
func Fused0115ReluAddExpName() string { return "relu_add_exp" }
