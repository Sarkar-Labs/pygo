package tensor

import "fmt"

func Fused0837CosMulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_mul_exp: nil operand")
	}
	r := x.Mul(y)
	r = r.Cos()
	r = r.Exp()
	return r, nil
}
func Fused0837CosMulExpName() string { return "cos_mul_exp" }
