package tensor

import "fmt"

func Fused0715LogAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Log10()
	return r, nil
}
func Fused0715LogAddLog10Name() string { return "log_add_log10" }
