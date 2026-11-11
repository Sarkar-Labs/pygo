package tensor

import "fmt"

func Fused0042CosMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	return r, nil
}
func Fused0042CosMulName() string { return "cos_mul" }
