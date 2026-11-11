package tensor

import "fmt"

func Fused0540ExpAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Exp()
	r = r.ReLU()
	return r, nil
}
func Fused0540ExpAddReluName() string { return "exp_add_relu" }
