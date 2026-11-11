package tensor

import "fmt"

func Fused0017AbsMul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	return r, nil
}
func Fused0017AbsMulName() string { return "abs_mul" }
