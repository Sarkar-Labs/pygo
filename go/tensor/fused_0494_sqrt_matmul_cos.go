package tensor

import "fmt"

func Fused0494SqrtMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Cos()
	return r, nil
}
func Fused0494SqrtMatmulCosName() string { return "sqrt_matmul_cos" }
