package tensor

import "fmt"

func Fused0402AbsMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Abs()
	r = r.Cos()
	return r, nil
}
func Fused0402AbsMulCosName() string { return "abs_mul_cos" }
