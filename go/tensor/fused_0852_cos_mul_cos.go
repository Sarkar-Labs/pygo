package tensor

import "fmt"

func Fused0852CosMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Cos()
	return r, nil
}
func Fused0852CosMulCosName() string { return "cos_mul_cos" }
