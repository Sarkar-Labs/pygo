package tensor

import "fmt"

func Fused1034CeilMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Cos()
	return r, nil
}
func Fused1034CeilMatmulCosName() string { return "ceil_matmul_cos" }
