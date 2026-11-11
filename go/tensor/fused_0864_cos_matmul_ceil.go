package tensor

import "fmt"

func Fused0864CosMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Ceil()
	return r, nil
}
func Fused0864CosMatmulCeilName() string { return "cos_matmul_ceil" }
