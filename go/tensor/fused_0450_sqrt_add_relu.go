package tensor

import "fmt"

func Fused0450SqrtAddRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_add_relu: nil operand")
	}
	r := x.Add(y)
	r = r.Sqrt()
	r = r.ReLU()
	return r, nil
}
func Fused0450SqrtAddReluName() string { return "sqrt_add_relu" }
