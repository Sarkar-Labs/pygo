package tensor

import "fmt"

func Fused1572Log2MulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Log2()
	r = r.Cos()
	return r, nil
}
func Fused1572Log2MulCosName() string { return "log2_mul_cos" }
