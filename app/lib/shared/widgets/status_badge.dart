import 'package:flutter/material.dart';
import '../../core/theme/app_colors.dart';

class StatusBadge extends StatelessWidget {
  final String status;
  final bool isSimulated;

  const StatusBadge({
    super.key,
    required this.status,
    this.isSimulated = false,
  });

  @override
  Widget build(BuildContext context) {
    Color bg;
    Color fg;
    String text = status.toUpperCase();

    switch (status.toLowerCase()) {
      case 'ready':
      case 'completed':
      case 'pass':
      case 'validated':
        bg = AppColors.success.withValues(alpha: 0.15);
        fg = AppColors.success;
        break;
      case 'executing':
      case 'running':
      case 'loading':
        bg = AppColors.primary.withValues(alpha: 0.15);
        fg = AppColors.primary;
        break;
      case 'checkpoint_missing':
      case 'not_loaded':
      case 'warning':
      case 'qualified_drift':
        bg = AppColors.warning.withValues(alpha: 0.15);
        fg = AppColors.warning;
        break;
      case 'offline':
      case 'failed':
      case 'error':
        bg = AppColors.error.withValues(alpha: 0.15);
        fg = AppColors.error;
        break;
      default:
        bg = AppColors.neutral.withValues(alpha: 0.15);
        fg = AppColors.neutral;
    }

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: fg.withValues(alpha: 0.3), width: 1),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: fg,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                text,
                style: TextStyle(
                  color: fg,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
        ),
        if (isSimulated) ...[
          const SizedBox(width: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
            decoration: BoxDecoration(
              color: Colors.deepOrange.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: Colors.deepOrange, width: 0.8),
            ),
            child: const Text(
              'SIMULATION',
              style: TextStyle(
                color: Colors.deepOrange,
                fontSize: 9,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ],
    );
  }
}
